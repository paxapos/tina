from django.core.management.base import BaseCommand, CommandError
from ia.ia_engine import IaEngine

class Command(BaseCommand):

    help = 'Admin IA traininig models'

    def add_arguments(self, parser):
        parser.add_argument('train', nargs='+', type=int)
        parser.add_argument('predict', nargs='+', type=int)
        pass

    def handle(self, *args, **options):
        engine = IaEngine()

        engine.train("Milanesas")
        prediction = engine.predict('C:/Users/fabbr/Tina/tina/training/pics/training/burned/pic_01.jpg')

        self.stdout.write(self.style.SUCCESS('PREDICTION ENDS "%s"' % prediction))
