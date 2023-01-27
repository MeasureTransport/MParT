import matlab.unittest.TestRunner
import matlab.unittest.TestSuite
import matlab.unittest.plugins.XMLPlugin

KokkosInitialize(2);

suite = testsuite
runner = TestRunner.withNoPlugins;
xmlFile = 'test-results-matlab.xml';
p = XMLPlugin.producingJUnitFormat(xmlFile);
runner.addPlugin(p)
results = runner.run(suite);
table(results)