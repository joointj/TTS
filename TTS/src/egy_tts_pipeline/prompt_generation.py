from __future__ import annotations

import random
from dataclasses import dataclass

from .config import GenerationConfig, NormalizationConfig
from .text_normalization import inspect_text_features, normalize_egyptian_text, preflight_tts_text
from .utils import utc_now_iso


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    domain: str
    intent: str
    template: str


PERSON_NAMES = ["أحمد", "مريم", "سارة", "كريم", "ندى", "شريف", "سلمى", "عمر", "منة", "حسام"]
DISTRICTS = [
    "مدينة نصر",
    "المعادي",
    "الدقي",
    "شبرا",
    "الهرم",
    "حلوان",
    "طنطا",
    "المنصورة",
    "إسكندرية",
    "الشيخ زايد",
]
LANDMARKS = [
    "محطة المترو",
    "المخبز",
    "الصيدلية",
    "الكوبري",
    "المول",
    "البنك",
    "الجامعة",
    "المستشفى",
    "السنتر",
]
FOODS = ["كشري", "فول", "طعمية", "عيش بلدي", "فراخ مشوية", "رز بلبن", "جبنة قديمة", "بطاطس"]
DRINKS = ["شاي", "قهوة", "عصير قصب", "ينسون", "مياه ساقعة"]
SERVICES = ["السباك", "الكهربائي", "فني التكييف", "عامل الغاز", "النجار"]
OBJECTS = ["الشاحن", "السماعة", "المفاتيح", "الريموت", "الكارت", "الموبايل"]
SHOP_TYPES = ["السوبر ماركت", "الصيدلية", "المخبز", "المكتبة", "محل الموبايلات", "الخضري"]
BORROWED_TERMS = ["الأبلكيشن", "الواي فاي", "الباسوورد", "الكود", "اللينك", "أونلاين"]
FAMILY_ROLES = ["ماما", "بابا", "أختي", "أخويا", "مراتي", "جوزي"]
WORK_ITEMS = ["الورقة", "الملف", "الفاتورة", "الإيميل", "الطلب", "التقرير"]
TIME_PHRASES = [
    "دلوقتي",
    "بكرة الصبح",
    "بعد الضهر",
    "بالليل",
    "آخر النهار",
    "بعد ساعة",
    "الساعة سبعة إلا ربع",
    "الساعة تمانية والنص",
]

PROMPT_TEMPLATES = [
    PromptTemplate("daily_01", "daily_life", "arrangement", "فكرني أعدي على {shop_type} وأنا راجع"),
    PromptTemplate("daily_02", "daily_life", "home", "معلش افتح الشباك شوية الجو حر أوي"),
    PromptTemplate("daily_03", "daily_life", "reminder", "قول لـ {person_name} يجيب {object_name} قبل ما يطلع"),
    PromptTemplate("daily_04", "daily_life", "coordination", "أنا تحت البيت استناني خمس دقايق"),
    PromptTemplate("transport_01", "transport", "ride_hailing", "أنا واقف عند {landmark} وعايز أوصل {district} بسرعة"),
    PromptTemplate("transport_02", "transport", "lost_item", "نسيت {object_name} في العربية امبارح"),
    PromptTemplate("transport_03", "transport", "navigation", "العنوان بعد {landmark} على طول جنب {shop_type}"),
    PromptTemplate("transport_04", "transport", "pickup", "خلي السواق يكلمني لما يوصل {district}"),
    PromptTemplate("shopping_01", "shopping", "order_food", "عايز {quantity_phrase} {food_item} من {shop_type}"),
    PromptTemplate("shopping_02", "shopping", "grocery", "هات {quantity_phrase} {drink_item} و{quantity_phrase} {food_item}"),
    PromptTemplate("shopping_03", "shopping", "order_adjustment", "خلي الطلب يطلع من غير شطة ومن غير بصل"),
    PromptTemplate("shopping_04", "shopping", "inventory", "لو سمحت شوف {shop_type} فاتح ولا قفل"),
    PromptTemplate("support_01", "customer_support", "internet_issue", "{borrowed_term} مش شغال من امبارح والنت بطيء جدا"),
    PromptTemplate("support_02", "customer_support", "access_issue", "ابعتلي {borrowed_term} تاني عشان الرسالة الأولى ما وصلتش"),
    PromptTemplate("support_03", "customer_support", "account_help", "أنا مش عارف أدخل على {borrowed_term} بعد التحديث"),
    PromptTemplate("support_04", "customer_support", "follow_up", "ممكن حد يكلمني بخصوص الشكوى النهارده"),
    PromptTemplate("family_01", "family_social", "coordination", "قول لـ {family_role} إني هتأخر شوية"),
    PromptTemplate("family_02", "family_social", "check_in", "اطمن على {person_name} وقولي وصل ولا لسه"),
    PromptTemplate("family_03", "family_social", "visit", "هنعدي على {person_name} بعد {time_phrase}"),
    PromptTemplate("family_04", "family_social", "planning", "لو {person_name} جه بدري عرفه يستناني"),
    PromptTemplate("work_01", "work_admin", "follow_up", "ابعت {work_item} لـ {person_name} قبل الاجتماع"),
    PromptTemplate("work_02", "work_admin", "scheduling", "أجل المعاد لـ {time_phrase} عشان عندي مشوار"),
    PromptTemplate("work_03", "work_admin", "approval", "راجع {work_item} ولما تخلص ابعتهولي"),
    PromptTemplate("work_04", "work_admin", "handoff", "خلي {person_name} يستلم {work_item} من الريسبشن"),
    PromptTemplate("home_01", "home_services", "maintenance", "ابعت {service_name} لـ {district} {time_phrase}"),
    PromptTemplate("home_02", "home_services", "maintenance", "التكييف بيطلع هوا بس ومش بيسقع"),
    PromptTemplate("home_03", "home_services", "building", "كلم البواب يفتح للـ {service_name}"),
    PromptTemplate("home_04", "home_services", "utilities", "فاتورة الكهربا شكلها عالية الشهر ده"),
    PromptTemplate("pay_01", "payments", "wallet_transfer", "حول {amount_phrase} جنيه لـ {person_name} على المحفظة"),
    PromptTemplate("pay_02", "payments", "cash", "أنا معايا فكة لحد {amount_phrase} جنيه"),
    PromptTemplate("pay_03", "payments", "invoice", "الفاتورة طالعة {amount_phrase} جنيه تقريبا"),
    PromptTemplate("pay_04", "payments", "collection", "فكرني أسحب فلوس قبل ما أعدي على {district}"),
]


def egyptian_number_words(number: int) -> str:
    if number < 0 or number > 999:
        raise ValueError("Supported range is 0..999.")
    units = {
        0: "صفر",
        1: "واحد",
        2: "اتنين",
        3: "تلاتة",
        4: "أربعة",
        5: "خمسة",
        6: "ستة",
        7: "سبعة",
        8: "تمانية",
        9: "تسعة",
        10: "عشرة",
        11: "حداشر",
        12: "اتناشر",
        13: "تلاتاشر",
        14: "أربعتاشر",
        15: "خمستاشر",
        16: "ستاشر",
        17: "سبعتاشر",
        18: "تمناشر",
        19: "تسعتاشر",
    }
    tens = {
        20: "عشرين",
        30: "تلاتين",
        40: "أربعين",
        50: "خمسين",
        60: "ستين",
        70: "سبعين",
        80: "تمانين",
        90: "تسعين",
    }
    if number < 20:
        return units[number]
    if number < 100:
        if number in tens:
            return tens[number]
        return f"{units[number % 10]} و{tens[(number // 10) * 10]}"
    hundreds_map = {100: "مية", 200: "ميتين", 300: "تلتمية", 400: "أربعمية", 500: "خمسمية", 600: "ستمية", 700: "سبعمية", 800: "تمانمية", 900: "تسعمية"}
    if number in hundreds_map:
        return hundreds_map[number]
    leading = hundreds_map[(number // 100) * 100]
    remainder = number % 100
    return f"{leading} و{egyptian_number_words(remainder)}"


def quantity_phrase(rng: random.Random) -> str:
    number = rng.choice([1, 2, 3, 4, 5, 6, 10, 12])
    return egyptian_number_words(number)


def amount_phrase(rng: random.Random) -> str:
    number = rng.choice([15, 20, 35, 50, 75, 100, 120, 150, 200, 250, 300, 450])
    return egyptian_number_words(number)


class PromptGenerator:
    def __init__(self, config: GenerationConfig, normalization: NormalizationConfig):
        self.config = config
        self.normalization = normalization
        self.rng = random.Random(config.seed)
        self.templates = [template for template in PROMPT_TEMPLATES if template.domain in config.domain_weights]
        self.template_groups: dict[str, list[PromptTemplate]] = {}
        for template in self.templates:
            self.template_groups.setdefault(template.domain, []).append(template)

    def _pick_domain(self) -> str:
        domains = list(self.config.domain_weights.keys())
        weights = [self.config.domain_weights[name] for name in domains]
        return self.rng.choices(domains, weights=weights, k=1)[0]

    def _render_template(self, template: PromptTemplate) -> str:
        fields = {
            "person_name": self.rng.choice(PERSON_NAMES),
            "district": self.rng.choice(DISTRICTS),
            "landmark": self.rng.choice(LANDMARKS),
            "food_item": self.rng.choice(FOODS),
            "drink_item": self.rng.choice(DRINKS),
            "service_name": self.rng.choice(SERVICES),
            "object_name": self.rng.choice(OBJECTS),
            "shop_type": self.rng.choice(SHOP_TYPES),
            "borrowed_term": self.rng.choice(BORROWED_TERMS),
            "family_role": self.rng.choice(FAMILY_ROLES),
            "work_item": self.rng.choice(WORK_ITEMS),
            "time_phrase": self.rng.choice(TIME_PHRASES),
            "quantity_phrase": quantity_phrase(self.rng),
            "amount_phrase": amount_phrase(self.rng),
        }
        return template.template.format(**fields)

    def generate(self, sample_ids: list[str], existing_texts: set[str] | None = None) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen = set(existing_texts or set())
        for sample_id in sample_ids:
            for _attempt in range(50):
                domain = self._pick_domain()
                template = self.rng.choice(self.template_groups[domain])
                raw_text = self._render_template(template)
                tts_text = preflight_tts_text(raw_text)
                normalized_text = normalize_egyptian_text(raw_text, self.normalization)
                if normalized_text in seen:
                    continue
                features = inspect_text_features(tts_text)
                if not self.config.allow_latin_tokens and bool(features["contains_latin"]):
                    continue
                if not self.config.allow_arabizi and bool(features["contains_digits"]):
                    continue
                word_count = int(features["word_count"])
                if word_count < self.config.min_words or word_count > self.config.max_words:
                    continue
                seen.add(normalized_text)
                timestamp = utc_now_iso()
                rows.append(
                    {
                        "id": sample_id,
                        "prompt_text": raw_text,
                        "tts_text": tts_text,
                        "normalized_text": normalized_text,
                        "approved_text": None,
                        "domain": domain,
                        "intent": template.intent,
                        "template_id": template.template_id,
                        "speaker_id": None,
                        "text_features": features,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }
                )
                break
            else:
                raise RuntimeError("Could not generate a unique prompt after multiple attempts.")
        return rows
