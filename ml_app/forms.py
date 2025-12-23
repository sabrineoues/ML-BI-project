from django import forms


class KNNSampleForm(forms.Form):
    salary = forms.FloatField(label='Salary')
    company_name_freq = forms.FloatField(label='Company Name')
    Location_freq = forms.CharField(label='Location')


class XGBForm(forms.Form):
    feature1 = forms.FloatField()
    feature2 = forms.FloatField()


class RegressionForm(forms.Form):
    experience = forms.FloatField()


class KMeansForm(forms.Form):
    skill = forms.FloatField()
    salary = forms.FloatField()
