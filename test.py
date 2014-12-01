str = "jozko"
print "Name: %s\n%s" % (str, 30 * ("-"))
list = ["String", 45, 45.44433, ["another", "list"], ("something", "weird")]
print list[:]

def passing(a, b, c):
	c = a+b
	if c == 9:
		return 10
	elif c == 15:
		return "patnast"
	else:
		return c 

a = 10
b = 5
c = 0

c = passing(a, b, c)
print "c is %s\n%s" % (c, 30 * ("-"))

class Foo(object):
	static = 10
	def __init__(self):
		self.var = 7.5
	
	def getVar(self):
		return self.var

	def addToVar(self, argv):
		self.var+=argv
		return self.var

foo = Foo()
print "foo var: %f\n%s" % (foo.getVar(), 30 * ("-"))
foo.addToVar(0.5)
print "foo var: %d\n%s" % (foo.getVar(), 30 * ("-"))
foo2 = Foo()
Foo.static = 50
print "foo static: %d\n%s" % (foo.static, 30 * ("-"))
foo.static = 40
print "foo static: %d\n%s" % (foo.static, 30 * ("-"))
print "foo2 static: %d\n%s" % (foo2.static, 30 * ("-"))
