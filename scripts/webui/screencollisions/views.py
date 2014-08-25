from collections import defaultdict

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count
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

def index(request):
    data = defaultdict(int, {})

    collisions = Collision.objects.prefetch_related('curatedanswer_set').all()
    for collision in collisions:
        answers = set(ca.answer for ca in collision.curatedanswer_set.all())
        if len(answers) > 1:
            data['xx'] += 1
        elif len(answers) < 1:
            data['un'] += 1
        else:
            data[answers.pop()] += 1

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
    #done = CuratedAnswer.objects.filter(curator=user).values_list('collision', flat=True)
    #return Collision.objects.order_by('experiment_id', 'blob_id').exclude(id__in=done).first()
    return Collision.objects.order_by('experiment_id', 'blob_id').filter(curatedanswer__isnull=True).first()

@login_required
def start(request):
    return redirect(next_to_screen(request.user))

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
        return redirect(next_to_screen(request.user))

def category(request, cat):
    collisions = Collision.objects.prefetch_related('curatedanswer_set').all()
    results = []
    for collision in collisions:
        answers = set(ca.answer for ca in collision.curatedanswer_set.all())
        if len(answers) == 1 and cat in FIELD_NAMES and answers.pop() == cat:
            results.append(collision)
        elif len(answers) > 1 and cat == 'xx':
            results.append(collision)
        elif len(answers) < 1 and cat == 'un':
            results.append(collision)

    results.sort(key=lambda c: '{}_{:05}'.format(c.experiment_id, c.blob_id))

    context = {
        'catname': FIELD_NAMES[cat],
        'collisions': results,
    }
    return render(request, 'screencollisions/catlist.html', context)
