from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        stock_symbol = request.form.get('stock_symbol')
        task = request.form.get('task')

        data = CustomData(stock_symbol=stock_symbol, task=task)
        symbol, task_type = data.get_inputs()

        pipeline = PredictPipeline()
        try:
            result = pipeline.predict(symbol, task_type)

            if task_type == 'classification':
                result = "Up 📈" if result == 1 else "Down 📉"
            elif task_type == 'regression':
                result = round(float(result), 2)  # just to make sure it's nicely formatted

            return render_template('home.html', results=result, task=task_type, stock=stock_symbol)
        
        except Exception as e:
            # You can render error message if something fails
            return render_template('home.html', results=f"Error: {str(e)}", task=task_type, stock=stock_symbol)

       

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

