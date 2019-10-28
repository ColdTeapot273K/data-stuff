#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

poi = 0
for i in enron_data:
    if enron_data.get(i).get('poi') == 1:
        poi += 1

print('length:', len(enron_data), 'poi:', poi)
print(enron_data['LAY KENNETH L'])
# print(enron_data['COLWELL WESLEY'])
# print(enron_data["SKILLING JEFFREY K"])

#########
people = ("SKILLING JEFFREY K", "LAY KENNETH L", "FASTOW ANDREW S")
who = ''
money = 0
for i in people:
    if money < enron_data[i]["total_payments"]:
        money = enron_data[i]["total_payments"]
        who = i


print(who)
# Wrong code^

salaries = 0
emails = 0
for i in enron_data:
    if enron_data.get(i).get('salary') != 'NaN':
        salaries += 1
    if enron_data.get(i).get('email_address') != 'NaN':
        emails += 1

print('{} salaries and {} emails available'.format(salaries, emails))

count = 0 + 10
tpayments = 0
for i in enron_data:
    count += 1
    if enron_data.get(i).get('total_payments') != 'NaN':
        tpayments += 1

print('{}% of total payments available from {} tpayments and {} folks'.format(tpayments/count,
                                                                              tpayments, count))

count = 0 + 10
tpayments = 0
for i in enron_data:
    if enron_data.get(i).get('poi') == 1:
        count += 1
        if enron_data.get(i).get('total_payments') != 'NaN':
            tpayments += 1

print('{}% of total payments for POI available'.format(tpayments/count))
print('{}% of total payments for POI available from {} tpayments and {} POI'.format(tpayments/count,
                                                                              tpayments, count))
