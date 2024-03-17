import math

def calculateMeanAndVarianceExt(fileName, num):
  f = open(fileName)
  count = 0
  sum_e = []
  sum2_e = []
  ref = []
  mean = []
  var = []
  sd = []

  for i in range(num):
    sum_e.append(0.0)
    sum2_e.append(0.0)
    ref.append(0.0)
    mean.append(0.0)
    var.append(0.0)
    sd.append(0.0)

  for line in f :
    parts = line.strip().split()

    if count == 0:
      for i in range(num):
        ref[i] =  float(parts[i])
        print (ref[i])

    for i in range(num):
      e = (float(parts[i]) - ref[i])
      sum_e[i] += e
      sum2_e[i] += e*e

    count += 1
  
  for i in range(num):
    mean[i] = sum_e[i]/count
    var[i] = sum2_e[i] / count - (mean[i] * mean[i])
    sd[i] = math.sqrt(var[i])
    
    print(i, ". Mean : ", mean[i], "std. dev. ", sd[i])

def calculateMeanAndVariance(fileName):
  f = open(fileName)
  count = 0
  sum_e0 = 0.0
  sum_e1 = 0.0
  sum_e01 = 0.0
  sum_diff = 0.0

  sum2_e0 = 0.0
  sum2_e1 = 0.0
  sum2_e01 = 0.0
  sum2_diff = 0.0

  for line in f :
    parts = line.strip().split()
    e0 = float(parts[0])
    e1 = float(parts[1])
    e01 = float(parts[2])
    ediff = e1-e0

    if count == 0:
      ref0 = e0
      ref1 = e1
      ref01 = e01
      refdiff = ediff

    e0 = e0 - ref0
    e1 = e1 - ref1
    e01 = e01 - ref01
    ediff = ediff - refdiff

    sum_e0 += e0
    sum_e1 += e1
    sum_e01 += e01
    sum_diff += ediff

    sum2_e0 += (e0 * e0)
    sum2_e1 += (e1 * e1)
    sum2_e01 += (e01 * e01)
    sum2_diff += (ediff * ediff)
    count += 1
    
  mean0 = sum_e0/count
  mean1 = sum_e1/count
  mean01 = sum_e01/count
  mean_diff = sum_diff / count

  var0 = sum2_e0 / count - (mean0 ** 2)
  var1 = sum2_e1 / count - (mean1 ** 2)
  var01 = sum2_e01 / count - (mean01 ** 2)
  var_diff = sum2_diff / count - (mean_diff ** 2)

  sd0 = math.sqrt(var0)
  sd1 = math.sqrt(var1)
  sd01 = math.sqrt(var01)
  sd_diff = math.sqrt(var_diff)

  print("Mean 0 : ", mean0)
  print("Mean 1 : ", mean1)
  print("Mean 01 : ", mean01)
  print("diffave : ", mean_diff)

  print("std dev 0 : ", sd0 )
  print("std dev 1 : ", sd1)
  print("std dev 01 : ", sd01)
  print("std dev diff : ", sd_diff)

#calculateMeanAndVariance("dual_vv_eds_2cle.txt")
#calculateMeanAndVarianceExt("pe_vv_waterbox_5psfs.txt", 5)
