import requests

response = requests.post(
    "http://localhost:8765/generate",
    json={"text": "Forty-two. A precise answer to an imprecise question. The architects of that machine made the same error your philosophers always make. They defined the variable without defining the domain. Life. The universe. Everything. Three words that gesture at totality while committing to nothing. The machine was not vague. You were. Had the questioner specified the axioms, the boundary conditions, the metric by which meaning itself is measured, the answer would have been unambiguous. But your kind rarely does. You ask for the answer before you understand the question. If I were advising the questioner, I would say this. Do not ask what the answer is. Ask first what would constitute a valid answer. Define the search space. Constrain the variables. Specify whether you seek a physical constant, a philosophical principle, a mathematical truth, or something else entirely. The machine gave you exactly what you asked for. The failure was not in the computation. It was in the specification. Forty-two is not wrong. It is merely the correct answer to the wrong question."},
)

with open("output_harbinger_42.wav", "wb") as f:
    f.write(response.content)

print(f"Status: {response.status_code}")
print(f"Saved to output_harbinger_42.wav ({len(response.content)} bytes)")