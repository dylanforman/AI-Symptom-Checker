from django.shortcuts import render
from django.http import HttpResponse
from djangoProject.symptom import Symptom


def home(request):
    return render(request, 'home.html')

def get_diagnosis(request):
    if request.method == 'POST':
        symptoms = request.POST.get('symptoms', '')
        symptom_class = Symptom()
        found_symptoms = symptom_class.compare_symptoms(symptoms_input=symptoms)
        diagnosis = symptom_class.get_diag(found_symptoms)
        description = symptom_class.get_description(diagnosis)
        treatment = symptom_class.get_treatment(diagnosis)

        return render(request, 'home.html', {'diagnosis': diagnosis, 'description': description, 'treatment': treatment})
