import HelperMethods

saved = HelperMethods.retrieve_history()
losses = []
accuraces = []
perplexities = []
meteors = []
val_losses = []
val_accuraces = []
val_perplexities = []
val_meteors = []
epoch_times = []
params = []
for save in saved:
  losses.append(saved[save]['loss'])
  accuraces.append(saved[save]['accuracy'])
  meteors.append(saved[save]['meteor'])
  val_accuraces.append(saved[save]['val_accuracy'])
  val_losses.append(saved[save]['val_loss'])
  val_meteors.append(saved[save]['val_meteor'])
  perplexities.append(saved[save]['perplexity'])
  val_perplexities.append(saved[save]['val_perplexity'])
  epoch_times.append(saved[save]['epoch_time'])
  params.append(saved[save]['param'])


  print(str(saved[save]['param'])+" train bleu: "+str(round(saved[save]['train_accuracy']*100,2)))
  print(str(saved[save]['param']) + " test bleu: " + str(round(saved[save]['test_accuracy']*100,2)))
  print(str(saved[save]['param'])+" train meteor: "+str(round(saved[save]['train_meteor']*100,2)))
  print(str(saved[save]['param']) + " test meteor: " + str(round(saved[save]['test_meteor']*100,2)))
  print(str(saved[save]['param'])+" train perplexity: "+str(round(saved[save]['train_perplexity'],2)))
  print(str(saved[save]['param']) + " test perplexity: " + str(round(saved[save]['test_perplexity'],2)))
  print(str(saved[save]['param']) + " time: " + str(saved[save]['time']))
  print("------------------------------------------")

HelperMethods.plot_result_multiple(losses,"Loss Comparison","Loss","Epoch",params)
HelperMethods.plot_result_multiple(accuraces,"Accuracy Comparison","Accuracy","Epoch",params)
HelperMethods.plot_result_multiple(perplexities,"Perplexity Comparison","Perplexity","Epoch",params)
HelperMethods.plot_result_multiple(meteors,"Meteor Comparison","Meteor","Epoch",params)
HelperMethods.plot_result_multiple(val_accuraces,"Val Accuracy Comparison","Accuracy","Epoch",params)
HelperMethods.plot_result_multiple(val_losses,"Val Loss Comparison","Loss","Epoch",params)
HelperMethods.plot_result_multiple(val_perplexities,"Val Perplexity Comparison","Perplexity","Epoch",params)
HelperMethods.plot_result_multiple(val_meteors,"Val Meteor Comparison","Meteor","Epoch",params)
HelperMethods.plot_result_multiple(epoch_times,"Epoch Time Comparison","Time","Epoch",params)