import os

env = Environment(ENV = os.environ, CC='icc', CXX='icpc')
env.Append(CPPFLAGS=['-xhost','-Wall','-qopenmp','-qopenmp-simd','-qopt-report', '-qopt-assume-safe-padding'])
env.Append(LINKFLAGS=['-qopenmp','-qopenmp-simd'])
env.Append(CPPPATH='/opt/intel/advisor/include')
conf = Configure(env)
if not conf.CheckLibWithHeader('m', 'math.h', 'c'):
    print 'Did not find libm'
if not conf.CheckLibWithHeader('iomp5', 'omp.h', 'c'):
    print 'Did not find libm'
if not conf.CheckLibWithHeader('memkind', 'hbwmalloc.h', 'c'):
    print 'Did not find libmemkind'
else:
    env.Append(CCFLAGS='-D__HBM__')
env = conf.Finish()
debug = ARGUMENTS.get('debug', 0)
if int(debug):
    env.Append(CCFLAGS=['-g','-O0'])
else:
    env.Append(CCFLAGS=['-g','-O2'])

env.Program('luDecomp', ['main.cc'])
