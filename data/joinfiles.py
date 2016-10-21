fout=open("out.csv","a")
# first files:

league_list = ['B1', 'D1', 'D2', 'E0', 'E1', 'E2', 'E3', 'EC', 'F1', 'F2', 'G1', 'I1', 'I2', 'N1', 'P1', 'SC0', 'SC1', 'SC2', 'SC3', 'SP1', 'SP2', 'T1']

base_address = "data "

for i in range(23):
    print "Starting league teams"
    for num2 in range (0,21):
        try:
            for league in league_list:
                f = open(base_address+str(i)+"/"+league+".csv")
                for line in f:
                    fout.write(line)
                    print "writing this:", line
        except IOError:
			print "File "+base_address+str(i)+"/"+league+".csv does not exist"

print "Ending league teams"
