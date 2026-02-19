from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    context = ""

    if request.method == "POST":
        query = request.form["query"]
        answer = f"You asked: {query}"

    return render_template("index.html", answer=answer, context=context)

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
