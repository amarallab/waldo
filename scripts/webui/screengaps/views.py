from collections import defaultdict

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import Gap, CuratedAnswer

def index(request):
    return render(request, 'screengaps/index.html', context)

def category(request):
    pass

def gap(request):
    pass

def start(request):
    pass
