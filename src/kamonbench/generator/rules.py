EXCLUDED_CONTAINERS = {'垂れ角', '長亀甲', '抱き角', '三つ盛り亀甲'}

MODS = ['覗き', '豆', '三つ盛り', '尻合せ三つ', '頭合せ三つ']
CONTAINERLESS_MODS = ['三つ盛り', '尻合せ三つ', '頭合せ三つ']

STACKING_PATTERNS = ['盛り', '尻合せ', '頭合せ']


def is_already_stacked(motif_name: str) -> bool:
    return any(pattern in motif_name for pattern in STACKING_PATTERNS)
