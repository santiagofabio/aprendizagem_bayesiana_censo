def aplica_label_encoder(x_base):
    from sklearn.preprocessing import LabelEncoder

    label_encoder_workclass =LabelEncoder()
    label_encoder_education =LabelEncoder()
    label_encoder_marital =LabelEncoder()
    label_encoder_occupation =LabelEncoder()
    label_encoder_relationship =LabelEncoder()
    label_encoder_race =LabelEncoder()
    label_encoder_sex =LabelEncoder()   
    label_encoder_country =LabelEncoder()

    x_base[:,1] =label_encoder_workclass.fit_transform(x_base[:,1])
    x_base[:,3] =label_encoder_education.fit_transform(x_base[:,3])
    x_base[:,5] =label_encoder_marital.fit_transform(x_base[:,5])
    x_base[:,6] =label_encoder_occupation.fit_transform(x_base[:,6] )
    x_base[:,7] =label_encoder_relationship.fit_transform(x_base[:,7])
    x_base[:,8] =label_encoder_race.fit_transform(x_base[:,8])
    x_base[:,9] =label_encoder_sex.fit_transform(x_base[:,9])
    x_base[:,13] =label_encoder_country.fit_transform(x_base[:,13])



    return(x_base)