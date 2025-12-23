from django import forms


class KNNSampleForm(forms.Form):
    company_name = forms.CharField(label="Company Name")
    location = forms.CharField(label="Location")
    salary = forms.FloatField(label="Salary")





class XGBForm(forms.Form):
    location = forms.CharField(label="Location")
    skill = forms.CharField(label="Skill")
    company_name = forms.CharField(label="Company Name")
    platform_name = forms.CharField(label="Platform Name")
    degree = forms.CharField(label="Degree")
    salary = forms.FloatField(label="Salary")  # si tu veux garder le salaire



class RegressionForm(forms.Form):
    experience = forms.FloatField()


class KMeansForm(forms.Form):
    skill = forms.FloatField()
    salary = forms.FloatField()
