# importing required packages

from django.shortcuts import render,redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from diab_retina_app import process
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.decorators import login_required

from multiclassificationDR import *

#create your views here
@login_required(login_url='login')

@csrf_exempt
def display(request):
    if request.method == 'GET':
        return render(request, 'index.html')

@csrf_exempt
def train(request):
    if request.method == 'GET':
        return render(request, 'train.html')
    else:
        try:
            runtrainingcode()
        except Exception as exp:
            pass
        return render(request, 'success.html')


@csrf_exempt
def process_image(request):
   # print('now in process image fun')
    if request.method == 'POST':
        img = request.POST.get('image')
        response = process.process_img(img)
        return HttpResponse(response, status=200)


def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')

        if pass1!=pass2:
            return HttpResponse("Your password and confrom password are not Same!!")
        else:

            my_user=User.objects.create_user(uname,email,pass1)
            my_user.save()
            return redirect('login')


    return render(request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('display')
        else:
            return HttpResponse ("Username or Password is incorrect!!!")

    return render (request,'login.html')



def LogoutPage(request):
    logout(request)
    return redirect('login')