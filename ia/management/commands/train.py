from django.core.management.base import BaseCommand, CommandError
from ia.ia_engine import IaEngine

class Command(BaseCommand):

    help = 'Admin IA traininig models'

    def add_arguments(self, parser):
        # parser.add_argument('anything', nargs='+', type=int)
        pass

    def handle(self, *args, **options):
        engine = IaEngine()

        engine.training_()
        prediction = engine.predict('Test_Pic')

        self.stdout.write(self.style.SUCCESS('PREDICTION ENDS "%s"' % prediction))
