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

    DISAGREE_ANSWER = 'disagreement'
    UNSCREENED_ANSWER = 'unscreened'

    class Meta:
        unique_together = ('experiment_id', 'collision_id')

    def __unicode__(self):
        return 'EID {}, {}->{}'.format(
                self.experiment_id, self.from_blob, self.to_blob)

    def get_absolute_url(self):
        return reverse('outcomes:outcome', kwargs={
                'eid': self.experiment_id,
                'from_blob': self.from_blob,
                'to_blob': self.to_blob,
            })

    def image_file(self):
        return self._image_file(self.experiment_id, self.collision_id)

    # for checking if the thing exists without actually having the model
    # object
    @staticmethod
    def _image_file(eid, target):
        return (settings.OUTCOME_IMAGES /
                '{}_{:05}.png'.format(eid, target))

    def answer_count(self):
        return CuratedAnswer.objects.filter(object_pk=self.id).count()

    @classmethod
    def category(cls, cat):
        if cat == cls.DISAGREE_ANSWER:
            objs = (cls.objects
                    .annotate(ans_values=models.Count('answers__answer', distinct=True))
                    .filter(ans_values__gt=1))
        elif cat == cls.UNSCREENED_ANSWER:
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
            return self.DISAGREE_ANSWER
        else:
            return self.UNSCREENED_ANSWER

    @staticmethod
    def fill():
        index = settings.OUTCOME_IMAGES / 'index.csv'
        with open(index, 'r') as f:
            next(f) # skip header
            for entry in f:
                eid, target, pA, pB, cA, cB = entry.split(',')

                if not self._image_file(eid, target).exists():
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
    )
    outcomes = models.ForeignKey(Outcome, related_name='answers')
    answer = models.CharField(max_length=60, choices=ANSWERS)
    curator = models.ForeignKey(User, editable=False, related_name='gap_answer_user')
    starred = models.BooleanField(default=False)

    class Meta:
        unique_together = ('gap', 'curator')

    def __unicode__(self):
        return 'EID {}, {}->{}, {} said {}{}'.format(
            self.gap.experiment_id, self.gap.from_blob, self.gap.to_blob,
            self.curator.username, self.answer, ' !!!' if self.starred else '')
