"""Canonical schema of the Mendeley retail dataset mgzvngzng2 v1 (Shahin, Swarna & Farid, 2025)."""

SHOP_FILE = "shop_info_with_geo.csv"
PRODUCT_FILE = "product_info.csv"
TARGET_FILE = "product_target_for_shop.csv"

SHOP_ID = "shop_code"

SHOP_COLUMNS = [
    "shop_code",
    "organizer",
    "place_nature",
    "place_name",
    "freq_type",
    "shop_type",
    "shop_style",
    "shop_importance",
    "ss_importance",
    "visit_target",
    "latitude",
    "longitude",
    "geometry",
    "zila",
    "upazila",
    "division",
]

SHOP_CATEGORICAL = [
    "organizer",
    "place_nature",
    "freq_type",
    "shop_type",
    "shop_style",
    "shop_importance",
    "ss_importance",
]

PRODUCT_COLUMNS = [
    "sku_id",
    "sku_name",
    "product_id",
    "product_name",
    "brand",
    "type",
    "packet_size",
    "price_a",
    "price_b",
    "price_c",
    "price_1",
    "price_2",
    "price_3",
    "ingredient",
    "freq",
    "in_packet",
    "total_volume",
    "total_price",
]

PRODUCT_TARGET_COLUMNS = [f"prod_{i}" for i in range(1, 18)]
TOTAL_TARGET = "Total"
TARGET_COLUMNS = [SHOP_ID, *PRODUCT_TARGET_COLUMNS, TOTAL_TARGET]

# Bangladesh bounding box (lat_min, lat_max, lon_min, lon_max) for GPS validation
BD_BBOX = (20.5, 26.7, 88.0, 92.7)

# As spelled in the dataset (pre-2018 romanizations: Barisal, Chittagong)
DIVISIONS = [
    "Barisal",
    "Chittagong",
    "Dhaka",
    "Khulna",
    "Mymensingh",
    "Rajshahi",
    "Rangpur",
    "Sylhet",
]
