from django.db import models
#from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User
from django.conf import settings
from django.core.urlresolvers import reverse

import pathlib
import re

class Gap(models.Model):
    experiment_id = models.CharField(max_length=20)
    from_blob = models.IntegerField()
    to_blob = models.IntegerField()

    class Meta:
        unique_together = ('experiment_id', 'from_blob', 'to_blob')

    def __unicode__(self):
        return 'EID {}, {}->{}'.format(
                self.experiment_id, self.from_blob, self.to_blob)

    def get_absolute_url(self):
        return reverse('gaps:gap', kwargs={
                'eid': self.experiment_id,
                'from_blob': self.from_blob,
                'to_blob': self.to_blob,
            })

    def image_file(self):
        return (settings.GAP_IMAGES /
                '{}_{:05}_{:05}.png'.format(
                    self.experiment_id, self.from_blob, self.to_blob))

    def answer_count(self):
        return CuratedAnswer.objects.filter(object_pk=self.id).count()

    @staticmethod
    def find_images():
        r = re.compile("(?P<eid>\d{8}_\d{6})_(?P<from>\d{5})_(?P<to>\d{5})\.png")
        for f in settings.GAP_IMAGES.glob('*.png'):
            rm = r.match(f.name)
            if not rm:
                continue

            eid, from_blob, to_blob = rm.groups()
            fields = {
                'experiment_id': eid,
                'from_blob': int(from_blob),
                'to_blob': int(to_blob),
            }
            yield fields


class CuratedAnswer(models.Model):
    ANSWERS = (
        ('valid', 'Valid connection'),
        ('invalid', 'Invalid connection'),
        ('unclear', 'Unclear/Other'),
    )
    gap = models.ForeignKey(Gap, related_name='answers')
    answer = models.CharField(max_length=60, choices=ANSWERS)
    curator = models.ForeignKey(User, editable=False, related_name='gap_answer_user')
    starred = models.BooleanField(default=False)

    class Meta:
        unique_together = ('gap', 'curator')

    def __unicode__(self):
        return 'EID {}, {}->{}, {} said {}{}'.format(
            self.gap.experiment_id, self.gap.from_blob, self.gap.to_blob,
            self.curator.username, self.answer, ' !!!' if self.starred else '')
