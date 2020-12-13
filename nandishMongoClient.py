from pymongo import MongoClient
import json
from bson import json_util
import datetime 
import  dateutil
from dateutil import parser
myclient = MongoClient('mongodb+srv://[URL]')

mydb = myclient["new"]

mycol = mydb["student_attention"]
mycol1= mydb["student_attention_updated"]
#print(myclient.list_database_names())
print(mycol)
Yawnc=0
Drowsyc=0
Hp=0
Sd=0
nt=0
fr=0
headleftc=0
headrightc=0
headdown=0      
output_date = datetime.datetime.now()
year = output_date.year
month = output_date.month
yesterday = output_date.day
dateStr = str(year) + "-" + str(month) + "-" + str(yesterday)
date = dateutil.parser.parse(dateStr)
cursor = mycol.find({'timestamp' : { '$gt' : date},'Regid':'RA1234'})
current_truth = "False"
for document in cursor:
        #  print(document['Regid'],document['yawn'],document['drowsy'],document['emotion'],document['headmove'])
        # print(type(document['yawn']))
        if document['yawn']=="True":
                # print(place)
                Yawnc+=1
        if document['drowsy']=="True":
            Drowsyc+=1        
        if document['emotion']=="happy":
            Hp+=1    
        if document['emotion']=="sad" :  
            Sd+=1 
        if document['emotion']=="neutral":   
            nt+=1    
        if document['emotion']=="fear" :  
            fr+=1      
        if document['headmove']=="Head down":
            headdown+=1
        if(document['headmove']=="Head left"):
            headleftc+=1
        if(document['headmove']=="Head right") :
            headrightc+=1

print(Yawnc)    
print(Drowsyc)            
data1={}
data1['Regid']="RA1234"
data1['yawn']=Yawnc
data1['drowsy']=Drowsyc
data1['happy']=Hp
data1['sad']=Sd
data1['neutral']=nt
data1['fear']=fr
data1['headdown']=headdown
data1['headleft']=headleftc
data1['headright']=headrightc
serializedMyData = json.dumps(data1, default=json_util.default)
x = mycol1.insert_one(data1)


def send_data(Regid,yawn,drowsy,emotion,headmotion,uuid,timestamp):
    data={}
    # data['ClientID'] = "ABC123"
    # data['Face_detected'] = True
    # data['Mask_detected'] = False
    # data['CameraID'] = '04'
    data["yawn"]=yawn
    data["drowsy"]=drowsy
    data["emotion"]=emotion
    data["headmove"]=headmotion
    data["uuid"]=uuid
    data["timestamp"]=timestamp
    data["Regid"]=Regid
    serializedMyData = json.dumps(data, default=json_util.default)
    x = mycol.insert_one(data)
    print(x.inserted_id)
now =(datetime.datetime.now())   
# print() 
# print(mycol.find_one())
# start = datetime.datetime(2020, 11, 10)
# end = datetime.datetime(2012, 11, 11)

# for doc in mycol.find({'timestamp': {'$gte': start, '$lt': end}}):
    # print (doc)
# for x in mycol.find({'Regid': 'RA1234'}):
#      print(x)
# print("done")      
