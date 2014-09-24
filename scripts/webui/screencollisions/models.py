from django.db import models
#from django.contrib.contenttypes.models import ContentType
from django.contrib.auth.models import User
from django.conf import settings
from django.core.urlresolvers import reverse

import pathlib
import re

class Collision(models.Model):
    experiment_id = models.CharField(max_length=20)
    blob_id = models.IntegerField()

    class Meta:
        unique_together = ('experiment_id', 'blob_id')

    def get_absolute_url(self):
        return reverse('sc:collision', kwargs={
                'eid': self.experiment_id,
                'bid': self.blob_id,
            })

    def image_file(self):
        return (settings.COLLISION_IMAGES /
                '{}_{:05}.png'.format(self.experiment_id, self.blob_id))

    def answer_count(self):
        return CuratedAnswer.objects.filter(object_pk=self.id).count()

    @staticmethod
    def find_images():
        r = re.compile("(?P<eid>\d{8}_\d{6})_(?P<bid>\d{5})\.png")
        for f in settings.COLLISION_IMAGES.glob('*.png'):
            rm = r.match(f.name)
            if not rm:
                continue

            eid, bid = rm.groups()
            yield eid, int(bid)


class CuratedAnswer(models.Model):
    ANSWERS = (
        ('00', 'No worms'),
        ('05', '>0, <1 worms'),
        ('10', '1 worm'),
        ('15', '>1, <2 worms'),
        ('20', '2 worms'),
        ('25', '>2, <3 worms'),
        ('30', '3 worms'),
        ('35', '>3 worms'),
    )
    collision = models.ForeignKey(Collision)
    answer = models.CharField(max_length=60, choices=ANSWERS)
    curator = models.ForeignKey(User, editable=False, related_name='collision_answer_user')
    starred = models.BooleanField(default=False)

    class Meta:
        unique_together = ('collision', 'curator')
