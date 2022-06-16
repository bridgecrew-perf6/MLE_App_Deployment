from Capstone import Prepare, Preprocess, Train, Predict, Dependencies
import logging

logging.basicConfig(filename='DL_Log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

file = "D:\MLE Capstone Project\Data\DL_data_prepared_multigroup.csv"""
train, valid, test, weights, bias = Prepare.prepare_data(file)
inputs, layers, train_ds, valid_ds, test_ds = Preprocess.preprocess_data(train, valid, test)
model = Train.make_model(inputs, layers, output_bias=bias)
model = Train.fit_model(model, train_ds, valid_ds, weights)
Predict.evaluate_model(model, test_ds)
yhat_valid = Predict.predict_from_model(model, valid_ds)
yhat_test = Predict.predict_from_model(model, test_ds)
Dependencies.make_dependencies(valid, test, yhat_valid, yhat_test)