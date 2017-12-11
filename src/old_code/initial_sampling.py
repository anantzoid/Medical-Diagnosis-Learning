import csv
from collections import Counter
def build_sample_notes():
    with open("/media/disk3/disk3/mimic3/DIAGNOSES_ICD.csv", "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=",")
        icd = {}
        for row in csvreader:
            icd["_".join([row[1], row[2]])] = row[-1]
    classes = [_[1] for _ in icd.iteritems()]
    classes = ["V290", "4019"]

    with open("/media/disk3/disk3/mimic3/NOTEEVENTS.csv", "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=",")
        data = []
        for row in csvreader:
            if row[2] == '':
                continue
            index = "_".join([row[1], row[2]])
            try:
                _class = icd[index]
                if _class in classes :
                    data.append([row[-1], icd[index]])
            except Exception, e:
                print str(e)
    with open("/media/disk3/disk3/notes_sample.csv", "wb") as csvf:
        csvwriter = csv.writer(csvf, delimiter=",")
        csvwriter.writerows(data)

def mean_encounter_per_patiens():
    with open("/media/disk3/disk3/mimic3/DIAGNOSES_ICD.csv", "rb") as csvf:
        csvreader = csv.reader(csvf, delimiter=",")
        subjects = {}
        for row in csvreader:
            try:
                _ = int(row[1])
            except:
                continue

            if subjects.get((row[1], row[2]), None):
                subjects[(row[1], row[2])] += 1
            else:
                subjects[(row[1], row[2])] = 1

    sorte_ = sorted(subjects.iteritems(), key=lambda (k,v): (v,k))
    print "Minimum adm:", sorte_[0]
    print "Maximum adm:", sorte_[-1]
    import numpy as np
    print "Avg. adm:", np.mean([_[1] for _ in sorte_])

    def slabbing(slab):
        count = 0
        for i in sorte_:
            if i[1] > slab[0] and i[1] <= slab[1]:
                count += 1
        return count

    slabs = [[1, 2],[2,5],[5,10],[10, 20],[20, 30], [30, 50], [50, 100], [100, 200], [200, 400], [400, 1000]]
    for slab in slabs:
        print "%d-%d: %d"%(slab[0], slab[1], slabbing(slab))
    
mean_encounter_per_patiens()
