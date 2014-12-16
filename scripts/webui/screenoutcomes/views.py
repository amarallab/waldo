import json
import itertools

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import Outcome, CuratedAnswer

from .manipulators import answer_group, next_to_screen, user_progress

FIELD_NAMES = {k: v for k, v in itertools.chain(
    CuratedAnswer.ANSWERS,
    (Outcome.DISAGREE_ANSWER,),
    (Outcome.UNSCREENED_ANSWER,),)
}

def index(request):
    data = {k: answer_group(Outcome, k).count() for k in FIELD_NAMES}

    data = [{'name': cv, 'number': data[ck], 'code': ck} for ck, cv in FIELD_NAMES.items()]
    data.sort(key=lambda x: x['number'], reverse=True)

    piedata = [('Screen Results', 'Results')]
    piedata.extend((d['name'], d['number']) for d in data)

    context = {
        'request': request,
        'data': data,
        'piejson': json.dumps(piedata),
    }

    if request.user.is_authenticated():
        context['user_progress'] = user_progress(request.user)

    return render(request, 'screenoutcomes/index.html', context)

def category(request, cat):
    results = (answer_group(Outcome, cat)
            .order_by('experiment_id', 'collision_id'))

    context = {
        'catname': FIELD_NAMES[cat],
        'outcomes': results,
    }

    if request.user.is_authenticated():
        context['user_progress'] = user_progress(request.user)

    return render(request, 'screenoutcomes/catlist.html', context)

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

        if request.user.is_authenticated():
            context['user_progress'] = user_progress(request.user)

        return render(request, 'screenoutcomes/outcome.html', context)

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

@login_required
def start(request):
    next_ = next_to_screen(request.user)
    return redirect(next_ if next_ else 'outcomes:index')
