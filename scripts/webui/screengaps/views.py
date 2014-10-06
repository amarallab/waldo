from collections import defaultdict
import json

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Count, Q, Max, Min
from django.http import HttpResponse
from django.shortcuts import render, redirect, get_object_or_404

from .models import Gap, CuratedAnswer

FIELD_NAMES = dict(CuratedAnswer.ANSWERS)
ANSWER_CHOICES = list(FIELD_NAMES)
FIELD_NAMES.update({
    'disagreement': 'Disagreement',
    'unscreened': 'Unscreened',
})

DISAGREE_ANSWER = 'disagreement'
UNSCREENED_ANSWER = 'unscreened'

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
        # objs = (model.objects
        #     .filter(answers__isnull=False)
        #     .exclude(~Q(answers__answer=key))
        #     .distinct())

    return objs

def index(request):
    data = {k: answer_group(Gap, k).count() for k in FIELD_NAMES}

    data = [{'name': cv, 'number': data[ck], 'code': ck} for ck, cv in FIELD_NAMES.items()]
    data.sort(key=lambda x: x['number'], reverse=True)

    piedata = [('Screen Results', 'Results')]
    piedata.extend((d['name'], d['number']) for d in data)

    context = {
        'request': request,
        'data': data,
        'piejson': json.dumps(piedata),
    }
    return render(request, 'screengaps/index.html', context)

def category(request, cat):
    results = (answer_group(Gap, cat)
            .order_by('experiment_id', 'from_blob', 'to_blob'))

    context = {
        'catname': FIELD_NAMES[cat],
        'gaps': results,
    }
    return render(request, 'screengaps/catlist.html', context)

@login_required
def start(request):
    next_ = next_to_screen(request.user)
    return redirect(next_ if next_ else 'gaps:index')

def next_to_screen(user):
    ordering = 'experiment_id', 'from_blob', 'to_blob'

    # see if any aren't screened at all (by any user)
    gap = (Gap.objects
            .order_by(*ordering)
            .filter(answers__isnull=True)
            .first())

    if gap is None:
        # if everything's done once, check what the user has left to do
        done = (CuratedAnswer.objects
                .filter(curator=user)
                .values_list('gap', flat=True))
        gap = (Gap.objects
                .order_by(*ordering)
                .exclude(id__in=done)
                .first())

    return gap

def gap(request, eid, from_blob, to_blob):
    gap = get_object_or_404(Gap, experiment_id=eid, from_blob=from_blob, to_blob=to_blob)

    if request.method == 'GET':
        # viewing
        context = {
            'gap': gap,
            'request': request,
        }
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
