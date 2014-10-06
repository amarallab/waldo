from collections import defaultdict

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Max, Min
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import Collision, CuratedAnswer

FIELD_NAMES = {
    '00': 'Not a worm',
    '05': '0.5 worms',
    '10': '1 worm',
    '15': '1.5 worms',
    '20': '2 worms',
    '25': '2.5 worms',
    '30': '3 worms',
    '35': '3.5 worms',
    'xx': 'Disagreement',
    'un': 'Unscreened',
}

DISAGREE_ANSWER = 'xx'
UNSCREENED_ANSWER = 'un'

def answer_group(model, key):
    if key == DISAGREE_ANSWER:
        objs = (model.objects
                .annotate(ans_values=Count('answers__answer', distinct=True))
                .filter(ans_values__gt=1))
    elif key == UNSCREENED_ANSWER:
        objs = model.objects.filter(answers__isnull=True)
    else:
        objs = (model.objects
            .filter(answers__isnull=False)
            .annotate(ans_max=Max('answers__answer'), ans_min=Min('answers__answer'))
            .filter(ans_max=key, ans_min=key))

    return objs

def index(request):
    data = {k: answer_group(Collision, k).count() for k in FIELD_NAMES}

    data = [{'name': cv, 'number': data[ck], 'code': ck} for ck, cv in FIELD_NAMES.items()]
    data.sort(key=lambda x: x['number'], reverse=True)

    piedata = [[d['name'], d['number']] for d in data]

    context = {
        'request': request,
        'data': data,
        'piedata': (','.join(repr(x) for x in piedata)),
    }

    return render(request, 'screencollisions/index.html', context)

def next_to_screen(user):
    done = CuratedAnswer.objects.filter(curator=user).values_list('collision', flat=True)
    return Collision.objects.order_by('experiment_id', 'blob_id').exclude(id__in=done).first()
    #return Collision.objects.order_by('experiment_id', 'blob_id').filter(answers__isnull=True, user__isnot=user).first()

@login_required
def start(request):
    next_ = next_to_screen(request.user)
    return redirect(next_ if next_ else 'collisions:index')

def collision(request, eid, bid):
    collision = get_object_or_404(Collision, experiment_id=eid, blob_id=bid)

    if request.method == 'GET':
        context = {
            'request': request,
            'eid': eid,
            'bid': bid,
            'image_file': collision.image_file(),
            'collision': collision,
        }
        return render(request, 'screencollisions/collision.html', context)

    elif request.method == 'POST' and request.user.is_authenticated():
        curans = CuratedAnswer.objects.get_or_create(
                collision=collision,
                curator=request.user,
            )[0]
        curans.answer = request.POST['answer']
        curans.starred = request.POST.get('on', 'off') == 'on'
        curans.save()
        next_ = next_to_screen(request.user)
        return redirect(next_ if next_ else 'collisions:index')

    else:
        return redirect('collisions:index')

def category(request, cat):
    results = (answer_group(Collision, cat)
            .order_by('experiment_id', 'blob_id'))

    context = {
        'catname': FIELD_NAMES[cat],
        'collisions': results,
    }
    return render(request, 'screencollisions/catlist.html', context)
