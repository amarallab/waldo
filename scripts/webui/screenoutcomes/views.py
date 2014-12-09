from collections import defaultdict

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Max, Min
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import Outcome, CuratedAnswer

FIELD_NAMES = CuratedAnswer.ANSWERS.copy()
FIELD_NAMES['xx'] = Outcome.DISAGREE_ANSWER
FIELD_NAMES['un'] = Outcome.UNSCREENED_ANSWER

def answer_group(model, key):
    if key == FIELD_NAMES['xx']:
        objs = (model.objects
                .annotate(ans_values=Count('answers__answer', distinct=True))
                .filter(ans_values__gt=1))
    elif key == FIELD_NAMES['un']:
        objs = model.objects.filter(answers__isnull=True)
    else:
        objs = (model.objects
            .filter(answers__isnull=False)
            .annotate(ans_max=Max('answers__answer'), ans_min=Min('answers__answer'))
            .filter(ans_max=key, ans_min=key))

    return objs

def index(request):
    data = {k: answer_group(Outcome, k).count() for k in FIELD_NAMES}

    data = [{'name': cv, 'number': data[ck], 'code': ck} for ck, cv in FIELD_NAMES.items()]
    data.sort(key=lambda x: x['number'], reverse=True)

    piedata = [[d['name'], d['number']] for d in data]

    context = {
        'request': request,
        'data': data,
        'piedata': (','.join(repr(x) for x in piedata)),
    }

    return render(request, 'screenoutcomes/index.html', context)

def next_to_screen(user):
    # one answer between all users
    #done = CuratedAnswer.objects.filter(curator=user).values_list('outcomes', flat=True)
    #return (Outcome.objects.order_by('experiment_id', 'collision_id').exclude(id__in=done).first())
    # each user answers all
    #return (Outcome.objects.order_by('experiment_id', 'collision_id').filter(answers__isnull=True, user__isnot=user).first())
    # each user answers all, random order (avoid fatigue/runs)
    return (Outcome.objects
        .filter(answers__isnull=True, user__isnot=user)
        .order_by('?')
        .first())

@login_required
def start(request):
    next_ = next_to_screen(request.user)
    return redirect(next_ if next_ else 'outcomes:index')

def outcome(request, eid, bid):
    outcome = get_object_or_404(Outcome, experiment_id=eid, collision_id=bid)

    if request.method == 'GET':
        context = {
            'request': request,
            'eid': eid,
            'bid': bid,
            'image_file': outcome.image_file(),
            'outcome': outcome,
        }
        return render(request, 'screenoutcome/outcome.html', context)

    elif request.method == 'POST' and request.user.is_authenticated():
        curans = CuratedAnswer.objects.get_or_create(
                outcome=outcome,
                curator=request.user,
            )[0]
        curans.answer = request.POST['answer']
        curans.starred = request.POST.get('on', 'off') == 'on'
        curans.save()
        next_ = next_to_screen(request.user)
        return redirect(next_ if next_ else 'outcomes:index')

    else:
        return redirect('outcomes:index')

def category(request, cat):
    results = (answer_group(Outcome, cat)
            .order_by('experiment_id', 'collision_id'))

    context = {
        'catname': FIELD_NAMES[cat],
        'outcomes': results,
    }
    return render(request, 'screenoutcomes/catlist.html', context)
