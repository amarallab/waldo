from __future__ import division
from collections import defaultdict
import json

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import Gap, CuratedAnswer

FIELD_NAMES = dict(CuratedAnswer.ANSWERS)
ANSWER_CHOICES = list(FIELD_NAMES)
FIELD_NAMES.update({
    Gap.DISAGREE_ANSWER: 'Disagreement',
    Gap.UNSCREENED_ANSWER: 'Unscreened',
})

DEFAULT_ORDER = 'experiment_id', 'from_blob', 'to_blob'

def index(request):
    data = {k: Gap.category(k).count() for k in FIELD_NAMES}

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

    return render(request, 'screengaps/index.html', context)

def category(request, cat):
    results = Gap.category(cat).order_by(*DEFAULT_ORDER)

    context = {
        'catname': FIELD_NAMES[cat],
        'gaps': results,
    }

    if request.user.is_authenticated():
        context['user_progress'] = user_progress(request.user)

    return render(request, 'screengaps/catlist.html', context)

@login_required
def start(request):
    next_ = next_to_screen(request.user)
    return redirect(next_ if next_ else 'gaps:index')

def next_to_screen(user):
    # see if any aren't screened at all (by any user)
    gap = (Gap.objects
            .order_by(*DEFAULT_ORDER)
            .filter(answers__isnull=True)
            .first())

    if gap is None:
        # if everything's done once, check what the user has left to do
        done = (CuratedAnswer.objects
                .filter(curator=user)
                .values_list('gap', flat=True))
        gap = (Gap.objects
                .order_by(*DEFAULT_ORDER)
                .exclude(id__in=done)
                .first())

    return gap

def user_progress(user):
    n_done = CuratedAnswer.objects.filter(curator=user).count()
    n_total = Gap.objects.count()

    return {
        'percent': 100 * (n_done / n_total),
        'done': n_done,
        'remaining': n_total - n_done,
        'total': n_total,
    }

def gap(request, eid, from_blob, to_blob):
    gap = get_object_or_404(Gap, experiment_id=eid, from_blob=from_blob, to_blob=to_blob)

    if request.method == 'GET':
        # viewing
        context = {
            'gap': gap,
            'request': request,
        }
        if request.user.is_authenticated():
            context['user_progress'] = user_progress(request.user)

        return render(request, 'screengaps/gap.html', context)

    elif request.method == 'POST' and request.user.is_authenticated():
        # returning an answer
        assert request.POST['answer'] in ANSWER_CHOICES, 'invalid answer code'

        curans = CuratedAnswer.objects.get_or_create(
                gap=gap,
                curator=request.user,
            )[0]
        curans.answer = request.POST['answer']
        curans.starred = request.POST.get('on', 'off') == 'on'
        curans.save()
        next_ = next_to_screen(request.user)
        return redirect(next_ or 'gaps:index')

    else:
        # silently ignore unauth'ed users trying to post
        return redirect('gaps:index')
