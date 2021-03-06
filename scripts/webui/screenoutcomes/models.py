from django.db import models
#from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User
from django.conf import settings
from django.core.urlresolvers import reverse

import pathlib

class Outcome(models.Model):
    experiment_id = models.CharField(max_length=20)
    collision_id = models.IntegerField()
    parent_a = models.IntegerField()
    parent_b = models.IntegerField()
    child_a = models.IntegerField()
    child_b = models.IntegerField()

    DISAGREE_ANSWER = ('disagreement', 'Disagreement')
    UNSCREENED_ANSWER = ('unscreened', 'Unscreened')

    class Meta:
        unique_together = ('experiment_id', 'collision_id')

    def __unicode__(self):
        return 'XID {}-{:05}'.format(
                self.experiment_id, self.collision_id)

    def get_absolute_url(self):
        return reverse('outcomes:outcome', kwargs={
                'eid': self.experiment_id,
                'bid': self.collision_id,
            })

    def image_file(self):
        return self._image_file(self.experiment_id, self.collision_id)

    # for checking if the thing exists without actually having the model
    # object
    @staticmethod
    def _image_file(eid, target):
        return (settings.OUTCOME_IMAGES / 'outcome_{}_{:05}.png'.format(eid, target))

    def answer_count(self):
        return CuratedAnswer.objects.filter(object_pk=self.id).count()

    @classmethod
    def category(cls, cat):
        if cat == cls.DISAGREE_ANSWER[0]:
            objs = (cls.objects
                    .annotate(ans_values=models.Count('answers__answer', distinct=True))
                    .filter(ans_values__gt=1))
        elif cat == cls.UNSCREENED_ANSWER[0]:
            objs = cls.objects.filter(answers__isnull=True)
        else:
            objs = (cls.objects
                .filter(answers__isnull=False)
                .annotate(ans_max=models.Max('answers__answer'),
                          ans_min=models.Min('answers__answer'))
                .filter(ans_max=cat, ans_min=cat))
            # attempting to use Q object; doesn't quite work
            # see http://stackoverflow.com/a/26209781/194586
            # objs = (cls.objects
            #     .filter(answers__isnull=False)
            #     .exclude(~models.Q(answers__answer=cat))
            #     .distinct())

        return objs

    def screen_result(self):
        answers = set(self.answers.all())
        n_answers = len(answers)
        if n_answers == 1:
            return answers.pop()
        elif n_answers > 1:
            return self.DISAGREE_ANSWER[0]
        else:
            return self.UNSCREENED_ANSWER[0]

    @classmethod
    def fill(cls):
        index = settings.OUTCOME_IMAGES / 'index.csv'
        with index.open() as f:
            next(f) # skip header
            for entry in f:
                data = iter(entry.split(','))
                eid = next(data)
                target, pA, pB, cA, cB = [int(x) for x in data]

                if not cls._image_file(eid, target).exists():
                    print('Could not find matching file for EID: {}, '
                          'target: {}'.format(eid, target))
                    continue

                fields = {
                    'experiment_id': eid,
                    'collision_id': target,
                    'parent_a': pA,
                    'parent_b': pB,
                    'child_a': cA,
                    'child_b': cB,
                }
                yield fields

class CuratedAnswer(models.Model):
    ANSWERS = (
        ('match', 'Matched (AA/BB)'),
        ('swap', 'Swapped (AB/BA)'),
        ('unclear', 'Unclear/Other'),
        ('badsegment', 'Bad Segmentation'),
    )
    outcome = models.ForeignKey(Outcome, related_name='answers')
    answer = models.CharField(max_length=60, choices=ANSWERS)
    curator = models.ForeignKey(User, editable=False, related_name='outcome_answer_user')
    starred = models.BooleanField(default=False)

    class Meta:
        unique_together = ('outcome', 'curator')

    def __unicode__(self):
        return '{}, {} said {}{}'.format(
            self.outcome, self.curator.username, self.answer,
            ' !!!' if self.starred else '')
