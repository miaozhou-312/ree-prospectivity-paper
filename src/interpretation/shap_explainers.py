"""
Section 13a -- SHAP explainers for RF (TreeSHAP) and DevNet
(DeepExplainer -> GradientExplainer -> KernelExplainer fallback chain).

`DevNet.forward()` returns a squeezed (N,) tensor -- fine for training, but
it breaks shap.DeepExplainer/GradientExplainer on some PyTorch/SHAP version
combinations. `_DevNetSHAPWrapper` calls `base_model.net(x)` directly,
skipping the squeeze so the output stays (N, 1), which both explainers
expect. This changes nothing about the model's actual predictions -- it's
purely a shape adapter for SHAP.

`explain_devnet` tries DeepExplainer first, then GradientExplainer, then
falls back to the fully model-agnostic (but much slower) KernelExplainer if
both gradient-based explainers raise. `explain_rf` uses TreeExplainer with
SHAP's default tree_path_dependent perturbation, which needs no explicit
background sample.
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import shap
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap'])
    import shap


class _DevNetSHAPWrapper(nn.Module):
    """Adapts a trained DevNet for SHAP: keeps the (N,1) output shape that
    DeepExplainer/GradientExplainer need instead of DevNet.forward()'s
    squeezed (N,) shape. Does not change what the model predicts."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model.net(x)


def explain_rf(rf_model, X_explain):
    """TreeSHAP for the RF production model. Returns (shap_values, expected_value)
    for the positive class, shape (n_explain, n_features)."""
    explainer = shap.TreeExplainer(rf_model)
    sv = np.array(explainer.shap_values(X_explain))
    ev = explainer.expected_value
    if sv.ndim == 3:            # (n, n_features, n_classes) -- keep positive class
        sv = sv[:, :, 1]
        ev = ev[1] if np.ndim(ev) > 0 else ev
    return sv, float(np.ravel(ev)[0])


def explain_devnet(model, X_bg, X_explain, device, base_seed):
    """DeepExplainer -> GradientExplainer -> KernelExplainer fallback chain
    for one DevNet model. Returns (shap_values, expected_value, method_name),
    shap_values shape (n_explain, n_features)."""
    torch.manual_seed(base_seed)
    wrapped = _DevNetSHAPWrapper(model).to(device).eval()
    X_bg_t = torch.tensor(X_bg, dtype=torch.float32, device=device)
    X_ex_t = torch.tensor(X_explain, dtype=torch.float32, device=device)

    for method_name, build in [
        ('DeepExplainer',     lambda: shap.DeepExplainer(wrapped, X_bg_t)),
        ('GradientExplainer', lambda: shap.GradientExplainer(wrapped, X_bg_t)),
    ]:
        try:
            explainer = build()
            sv = np.array(explainer.shap_values(X_ex_t))
            if sv.ndim == 3:
                sv = sv[..., 0]
            ev = float(np.ravel(explainer.expected_value)[0])
            return sv, ev, method_name
        except Exception as e:
            print(f'    {method_name} failed ({type(e).__name__}: {str(e)[:150]}), falling back ...')

    print('    using KernelExplainer (model-agnostic, slower) ...')

    def _f(x):
        wrapped.eval()
        with torch.no_grad():
            out = wrapped(torch.tensor(x, dtype=torch.float32, device=device))
        return out.cpu().numpy().reshape(-1)

    bg_summary = shap.kmeans(X_bg, min(10, len(X_bg)))
    explainer = shap.KernelExplainer(_f, bg_summary)
    sv = np.array(explainer.shap_values(X_explain, nsamples=100))
    if sv.ndim == 3:
        sv = sv[..., 0]
    ev = float(np.ravel(explainer.expected_value)[0])
    return sv, ev, 'KernelExplainer'
