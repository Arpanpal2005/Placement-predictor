from sklearn.linear_model import LogisticRegression
sc=LogisticRegression()
import pandas as pd
csv=pd.read_csv("C:\\Users\\palar\\OneDrive\\Desktop\\dataset.zip")
csv["Placement"] = csv["Placement"].map({"Yes": 1, "No": 0})
csv["Internship_Experience"] = csv["Internship_Experience"].map({"Yes": 1, "No": 0})
p = csv[['IQ', 'CGPA','Academic_Performance' ,'Internship_Experience','Extra_Curricular_Score', 'Projects_Completed']]  
l=csv['Placement']
sc.fit(p,l)
iq=int(input("Enter the IQ of the student:"))
cgp=float(input("Enter the CGPA of the student:"))
a=int(input("Enter the Academic performence of the student out of 10:"))
i=int(input("Enter Internship done by of the student(1) or not(0):"))
ex=int(input("Enter Extra_Curricular_Score"))

p=int(input("Enter the No. of project done by the student:"))
prob=sc.predict_proba([[iq,cgp, a, i,ex, p]])
predictt=sc.predict([[iq,cgp, a, i,ex, p]])[0]
if predictt==1:
    print("The student might get placement with a probability of",round(prob[0][1]*100,2),"%")
else:
    print("The student might not get placement with a probability of",round(prob[0][0]*100,2),"%")
