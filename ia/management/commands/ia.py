from django.core.management.base import BaseCommand, CommandError
from ia.ia_engine import IaEngine

class Command(BaseCommand):

    help = 'Admin IA traininig models'

    def add_arguments(self, parser):
        parser.add_argument('train', nargs='+', type=int)
        parser.add_argument('predict', nargs='+', type=int)
	pass


    def HandleTrain(self, *args, **options):
        engine = IaEngine()
	product = input("¿what do you want to train?")
	ruta = /tina/training/products 
	for	product	in ruta :
		try:
			engine.train("product")
		except product.DoesNotExist:
			raise CommandError('product "%s" does not exist' % ruta)
		product.opened = False
		product.save()


self.stdout.write(self.style.SUCCESS('Successfully closed train "%s"' % ruta))
	
    
    def HandlePredict(self, *args, **options):
      img = input("which image do you want to predict?")
      img_input = img_to_array(img.resize((150, 150)))
      array = model.predict(img_input)
      score = array[0]

      return score


self.stdout.write(self.style.SUCCESS('PREDICTION ENDS "%s"' % prediction))
