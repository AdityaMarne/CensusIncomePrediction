from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data = CustomData(
            age=float(request.form.get('age')),# type: ignore
            education_num = float(request.form.get('education_num')),# type: ignore
            capital_gain = float(request.form.get('capital_gain')),# type: ignore
            hours_per_week = float(request.form.get('hours_per_week')),# type: ignore
            workclass = request.form.get('workclass'),# type: ignore
            education= request.form.get('education'),# type: ignore
            marital_status = request.form.get('marital_status'),# type: ignore
            occupation = request.form.get('occupation'),# type: ignore
            relationship = request.form.get('relationship'),# type: ignore
            race = request.form.get('race'),# type: ignore
            sex = request.form.get('sex'),# type: ignore
            native_country=request.form.get('native_country') # type: ignore
        )

        final_new_data=data.get_data_as_dataframe()
        #print(final_new_data)
        #print(request.form)
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result = pred[0]
        output =""
        if result == 0:
            output = "Income is Less than 50K"
        else:
            output = "Income is More than 50K"
        return render_template('index.html',final_result=output)




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
