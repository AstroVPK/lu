import os

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

env = Environment(ENV = os.environ, CC='icc', CXX='icpc')
env.Append(CPPFLAGS=['-xhost','-Wall','-qopenmp','-qopenmp-simd','-qopt-report', '-qopt-assume-safe-padding'])
env.Append(LINKFLAGS=['-qopenmp','-qopenmp-simd'])
env.Append(CPPPATH='/home/AstroVPK/intel/advisor/include')
conf = Configure(env)
HBM = ARGUMENTS.get('HBM', "True")
if not conf.CheckLibWithHeader('m', 'math.h', 'c'):
    print 'Did not find libm'
if not conf.CheckLibWithHeader('iomp5', 'omp.h', 'c'):
    print 'Did not find libm'
if not conf.CheckLibWithHeader('memkind', 'hbwmalloc.h', 'c'):
    print 'Did not find libmemkind'
else:
    if str2bool(HBM):
        env.Append(CCFLAGS='-D__HBM__')
env = conf.Finish()
debug = ARGUMENTS.get('debug', "False")
advisor = ARGUMENTS.get('advisor', "False")
assembly = ARGUMENTS.get('assembly', "False")
if str2bool(debug):
    env.Append(CCFLAGS=['-g','-O0'])
else:
    if str2bool(advisor):
        env.Append(CCFLAGS=['-g','-O2'])
    else:
        env.Append(CCFLAGS=['-O3'])
if str2bool(assembly):
    env.Append(CCFLAGS=['-save-temps']) #,'-ip-no-inlining','-ip-no-pinlining'])
env.Program('luDecomp', ['main.cc'])
