from django.core.management.base import BaseCommand, CommandError
from ia.ia_engine import IaEngine

class Command(BaseCommand):

    help = 'Admin IA traininig models'

    def add_arguments(self, parser):
        parser.add_argument('productName', nargs=1, type=str)


    def handle(self, *args, **options):
        engine = IaEngine()
        ret = engine.train(options['productName'][0])

        self.stdout.write(self.style.SUCCESS('Successfully closed train "%s"' % ret))
        

    



