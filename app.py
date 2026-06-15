from src.pipeline import analyze

while True:

    text = input(
        "\nEnter Financial Statement: "
    )

    result = analyze(text)

    print(
        "\nSentiment:",
        result["sentiment"]
    )

    print(
        "\nConfidence:",
        result["confidence"]
    )

    print(
        "\nExplanation:",
        result["explanation"]
    )
    choice = input("\nDo you want to analyze another statement? (y/n): ").lower()
    if choice != 'y':
        print("\nExiting...")
        break