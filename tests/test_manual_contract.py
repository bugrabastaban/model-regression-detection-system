
from pydantic import ValidationError
from contracts import ClassifierOutput

# Geçerli çıktı
valid = ClassifierOutput(
    category="billing",
    summary="The customer reports being charged twice."
)

print("Geçerli çıktı:")
print(valid.model_dump())

# Geçersiz kategori reddedilmeli
try:
    ClassifierOutput(
        category="other",
        summary="The customer needs help."
    )
    raise AssertionError("Geçersiz kategori kabul edildi!")
except ValidationError:
    print("\nGeçersiz kategori doğru şekilde reddedildi.")

# Boş özet reddedilmeli
try:
    ClassifierOutput(
        category="technical",
        summary="   "
    )
    raise AssertionError("Boş özet kabul edildi!")
except ValidationError:
    print("Boş özet doğru şekilde reddedildi.")

print("\nPydantic sözleşme testi başarılı!")
