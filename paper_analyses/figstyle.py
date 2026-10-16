"""Human-readable feature labels and shared colours for paper figures."""

MODEL_COLORS = {'DevNet': '#d03b3b', 'RF': '#2a78d6', 'BaggingPU': '#3a9e5c', 'nnPU': '#8a5cc2'}

_EXACT = {
    'Canada_BA': 'Bouguer anomaly (BA)',
    'Canada_BA_1VD': 'BA 1VD',
    'Canada_BA_THD': 'BA THD',
    'Canada_BA_THD_Tilt': 'BA THD tilt',
    'Canada_BA_HP40km': 'BA high-pass 40 km',
    'Canada_BA_HP100km': 'BA high-pass 100 km',
    'TMI': 'TMI',
    'DRTP': 'DRTP',
    'DRTP_1VD': 'DRTP 1VD',
    'DRTP_THD': 'DRTP THD',
    'DRTP_THD_Tilt': 'DRTP THD tilt',
    'GeophysicsGravity_Up30km_USCanada': 'Gravity up-cont. 30 km',
    'GeophysicsGravity_Up30km_HGM_USCanada': 'Gravity up-cont. 30 km HGM',
    'GeophysicsLAB_USCanada': 'LAB depth',
    'USCanada_Moho': 'Moho depth',
    'USCanadaMagRTP_DeepSources': 'Deep-source RTP',
    'USCanadaMagRTP_HGMDeepSources': 'Deep-source RTP HGM',
    'AncientPassiveMargins_proximity': 'Passive-margin proximity',
    'Faults_Canada_proximity': 'Fault proximity',
    'worm_DeepGravitySources_Worms_USCanada_proximity': 'Deep gravity-worm proximity',
    'worm_ShallowGravitySources_Worms_USCanada_proximity': 'Shallow gravity-worm proximity',
    'worm_DeepMagSources_Worms_USCanada_proximity': 'Deep magnetic-worm proximity',
    'worm_ShallowMagSources_Worms_USCanada_proximity': 'Shallow magnetic-worm proximity',
}


def pretty(col):
    if col in _EXACT:
        return _EXACT[col]
    if col.startswith('geo_era_'):
        return f"Era: {col[len('geo_era_'):]}"
    if col.startswith('lith_'):
        return f"Lith.: {col[len('lith_'):].replace('_', ' ')}"
    if '_glcm_' in col:
        base, tex = col.split('_glcm_')
        return f"{_EXACT.get(base, base)} GLCM {tex}"
    return col
