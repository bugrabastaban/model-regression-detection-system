"""Manual live API check; importing during discovery must be offline."""

if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    from app.email_classifier import classify_email
    from app.contracts import ClassifierOutput

    email = "I was charged twice this month."

    result = classify_email(email)

    print("Model çıktısı:")
    print(result)

    # Fonksiyonun sözlük döndürmeye devam ettiğini doğrula
    assert isinstance(result, dict)

    # Dönen verinin sözleşmeye uyduğunu doğrula
    validated = ClassifierOutput.model_validate(result)

    assert validated.category == "billing"
    assert validated.summary.strip()

    print("\nGerçek API ve çıktı sözleşmesi testi başarılı!")
