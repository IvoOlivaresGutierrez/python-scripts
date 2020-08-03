# Ejemplo de clase Alumno con su iniciador y metodos

class Alumno:
	#Inicializamos los atributos
	def __init__(self, nombre, nota):
		self.nombre = nombre
		self.nota = nota

	def imprimir(self):
		print("alumno nombre: ",self.nombre)
		print("nota: ",self.nota)

	def resultados(self):
		if self.nota > 4:
			print ("Alumno aprobado")
		else:
			print("Alumno reprobado")

alumno1 = Alumno(nombre="Ivo", nota=4)
alumno2 = Alumno(nombre="Diego", nota=5)
alumno3 = Alumno(nombre="Sofia", nota=6)
alumno4 = Alumno(nombre="Andrea", nota=7)

alumno1.imprimir()
alumno1.resultados()
