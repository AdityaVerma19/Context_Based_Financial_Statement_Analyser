from pipeline import analyze

texts = [
    "Nvidia reported record quarterly revenue",
    "The company suffered heavy losses",
    "The company maintained previous guidance",
    "Production costs have declined across several facilities"
]

for t in texts:
    print(f"Statement: {t}")
    print(f"Result: {analyze(t)}")
    print("-" * 50)