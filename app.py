import streamlit as st
from streamlit_embedcode import github_gist
import numpy as np
import pandas as pd
import pickle

st.title("Prediction of Life Expectancy from Socio-Economic & Health Factors")


st.sidebar.header('  ')


menu = ["About the App", "Predict Life Expectancy", "Model Performance Metrics"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "About the App":
    
    option = ["What it Does?", "About the Dataset", "Data Preprocessing", 'Exploratory Data Analysis', 'Model Development & Evaluation']
    select = st.selectbox("Menu", option)
    if select == "What it Does?":
        st.subheader("About")
        st.markdown('WHO and UN created a dataset of the social, economic, and health-related status of 193 countries over the period  2000-2015. The dataset includes statistics on life expectancy, adult mortality, and more. We want to see how the predictors are related to each other and also with the target variable- human life expectancy. We also will predict life expectancy using a machine learning algorithm. ')

    elif select == "About the Dataset":
        st.subheader("Context")
        st.markdown('Although there have been a lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition, and mortality rates. It was found that the effect of immunization and human development index was not taken into account in the past. Also, some of the past research was done considering multiple linear regression based on the data set of one year for all the countries. Hence, this gives the motivation to resolve both the factors stated previously by formulating a regression model based on the mixed-effects model and multiple linear regression while considering data from a period of 2000 to 2015 for all the countries. Important immunization like Hepatitis B, Polio, and Diphtheria will also be considered. In a nutshell, this study will focus on immunization factors, mortality factors, economic factors, social factors, and other health-related factors as well. Since the observations this dataset is based on different countries, it will be easier for a country to determine the predicting factor which is contributing to lower the value of life expectancy. This will help in suggesting a country in which areas should be given importance to efficiently improve the life expectancy of its population.')
        st.subheader("Content")
        st.markdown("The project relies on the accuracy of the data. The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status as well as many other related factors for all countries The data-sets are made available to the public for health data analysis. The data-set related to life expectancy, health factors for 193 countries have been collected from the same WHO data repository website, and its corresponding economic data was collected from the United Nations website. Among all categories of health-related factors, only those critical factors were chosen which are more representative. It has been observed that in the past 15 years, there has been a huge development in the health sector resulting in the improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project, we have considered data from the year 2000-2015 for 193 countries for further analysis. The individual data files have been merged into a single data-set. On initial visual inspection of the data showed some missing values. As the data-sets were from WHO, we found no evident errors. Missing data were handled in R software by using the Missmap command. The result indicated that most of the missing data were for population, Hepatitis B, and GDP. The missing data were from less known countries like Vanuatu, Tonga, Togo, Cabo Verde, etc. Finding all data for these countries was difficult and hence, it was decided that we exclude these countries from the final model data-set. The final merged file(final dataset) consists of 22 Columns and 2938 rows which meant 20 predicting variables. All predicting variables were then divided into several broad categories:​Immunization related factors, Mortality factors, Economical factors, and Social factors.")
        st.subheader("Link to Dataset")
        st.markdown('More info can be found on kaggle. https://www.kaggle.com/kumarajarshi/life-expectancy-who')

    elif select == "Data Preprocessing":
        st.subheader("Major Steps")
        st.markdown('1. Data Cleaning')
        st.markdown('2. Handling Missing Values: Random Sample Imputation')
        st.markdown('3. Ordinal and Nominal Encoding of the categorical columns: df.status and df.countries')

        github_gist("https://gist.github.com/SumaiaParveen/33245edb50afe44a54dd1f008b80cbd7")

    elif select == 'Exploratory Data Analysis':
        st.subheader("Govt's Healthcare Expenditure Rate and People's Life Expectancy")
        from PIL import Image
        img = Image.open("Image/exp_le.png")
        st.image(img, width=800)

        st.markdown("1. A large portion of the dataset suggests that approximately 10% of expense implies an average life expectancy of 80 years.")
        st.markdown("2. Couple of 90 years life expectancy are observed where the healthcare expense rates are ~1% and ~8.5%")
        st.markdown("3. Bimodal distribution-- most of the data are primarily concentrated near expense rate = 10%, life expectancy = 80 and another set of data are seen where expense rate = 2.5%, life expectancy = 82.")

        st.subheader("Realtionship of Life Expectancy and Human Development Index, GDP and Education")
        from PIL import Image
        img = Image.open("Image/gdp_sc_le.png")
        st.image(img, width=700)

        st.markdown("1. More years spent in the school implies higher life expectancy.")
        st.markdown("2. GDP of the people/countries are mostly distributed around USD 1000.")
        st.markdown("3. Some people spent around 24 years in the school.")

        st.subheader("Relationship of Life Expectancy with Mortality Rates")
        from PIL import Image
        img = Image.open("Image/mort.png")
        st.image(img, width=800)

        st.markdown("1. Apparently, high mortality rate implies poor healthcare system and thus shorter life expectancy.")

        st.subheader("Relationship of Life Expectancy with Immunization Coverage")
        from PIL import Image
        img = Image.open("Image/imm.png")
        st.image(img, width=800)

        st.markdown("1. Low rate of immunization coverages implies shorter life expectancy and vice versa.")

        st.subheader("BMI vs Life Expectancy")
        from PIL import Image
        img = Image.open("Image/bmi.png")
        st.image(img, width=500)

        st.markdown("1. Most of the people have BMI around 55 and their apprximate life expectancy is 75 years.")
        st.markdown("2. A smaller group of people have BMI between (0-30) and their apprximate life expectancy is 62 years.")

        st.subheader("Average Life Expectancy of the Developing and Develpoed Countries")
        from PIL import Image
        img = Image.open("Image/stat.png")
        st.image(img, width=500)

        st.markdown("1. The people of developed countries may live approximately 10 years longer than the people from developing countries.")
        
        st.subheader("Life Expectancy and Alcohol Consumption")
        from PIL import Image
        img = Image.open("Image/alco.png")
        st.image(img, width=800)

        st.markdown("1. Higher life expectancy (almost 90 years long) with average 8.5 liters of pure alcohol.")
        st.markdown("2. A number of people drink less than 2.5 liters of alcohol and their average life expectancy is approximately 72 years.")

        #github_gist("https://gist.github.com/SumaiaParveen/1c765703a41ace8a19e3bd3ad2cd7cd0")

    elif select == 'Model Development & Evaluation':

        st.subheader("Model Selection")
        # Images
        from PIL import Image
        img = Image.open("Image/mod_sel.png")
        st.image(img, caption="LazyRegressor Result")

        st.subheader("Model Development: Optimized ExtraTressRegressor")
        st.markdown("Hyperparameters are optimized using RandomizedSearchCV. The best parameters are follows:")
        img = Image.open("Image/best_p.png")
        st.image(img, caption="Best Parameters found from RandomizedSearchCV Training")

        
        st.subheader("Model Evaluation")

        st.markdown('Mean absolute error (MAE): 1.0564712470881465') 
        st.markdown('Mean squared error (MSE): 3.2797066467789717')
        st.markdown('Root mean squared error (RMSE): 1.8109960372068659')
        st.markdown('R Squared: 0.9648817803845204')
        st.markdown('Adjusted R Squared: 0.9640212405806405')
        st.markdown('Accuracy (Computed from Actual & Predicted-- please see code**): 98.368%')

        img = Image.open("Image/dist_ev.png")
        st.image(img,  width=600, caption="Distribution Plot")

        img = Image.open("Image/residuals.png")
        st.image(img,  width=600, caption="Residual Plot")

        img = Image.open("Image/err.png")
        st.image(img,  width=600, caption="Error Plot")

        github_gist("https://gist.github.com/SumaiaParveen/f89e2f087a5c74cfe4d8e83bc90d2a76")

    else:
        st.markdown('          ')

elif choice == 'Predict Life Expectancy':

    if st.subheader("Country Name"):
        menu = ['Afghanistan', 'Albania', 'Algeria', 'Angola',
            'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia',
            'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
            'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
            'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina',
            'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria',
            'Burkina Faso', 'Burundi', "Côte d'Ivoire", 'Cabo Verde',
            'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
            'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
            'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
            'Czechia', "Democratic People's Republic of Korea",
            'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
            'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
            'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
            'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
            'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala',
            'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
            'Hungary', 'Iceland', 'India', 'Indonesia',
            'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy',
            'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
            'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
            'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
            'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
            'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
            'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia',
            'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
            'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
            'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 'Pakistan', 'Palau',
            'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
            'Poland', 'Portugal', 'Qatar', 'Republic of Korea',
            'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda',
            'Saint Kitts and Nevis', 'Saint Lucia',
            'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
            'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
            'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
            'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
            'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
            'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
            'The former Yugoslav republic of Macedonia', 'Timor-Leste', 'Togo',
            'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
            'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
            'United Arab Emirates',
            'United Kingdom of Great Britain and Northern Ireland',
            'United Republic of Tanzania', 'United States of America',
            'Uruguay', 'Uzbekistan', 'Vanuatu',
            'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
            'Zambia', 'Zimbabwe']
        
        choice = st.selectbox("Menu", menu)
        if choice == "Afghanistan" : 
            country =  0
        elif choice == "Albania" : 
            country =  1
        elif choice == "Algeria" : 
            country =  2
        elif choice == "Angola" : 
            country =  3
        elif choice == "Antigua and Barbuda" : 
            country =  4
        elif choice == "Argentina" : 
            country =  5
        elif choice == "Armenia" : 
            country =  6
        elif choice == "Australia" : 
            country =  7
        elif choice == "Austria" : 
            country =  8
        elif choice == "Azerbaijan" : 
            country =  9
        elif choice == "Bahamas" : 
            country =  10
        elif choice == "Bahrain" : 
            country =  11
        elif choice == "Bangladesh" : 
            country =  12
        elif choice == "Barbados" : 
            country =  13
        elif choice == "Belarus" : 
            country =  14
        elif choice == "Belgium" : 
            country =  15
        elif choice == "Belize" : 
            country =  16
        elif choice == "Benin" : 
            country =  17
        elif choice == "Bhutan" : 
            country =  18
        elif choice == "Bolivia (Plurinational State of)" : 
            country =  19
        elif choice == "Bosnia and Herzegovina" : 
            country =  20
        elif choice == "Botswana" : 
            country =  21
        elif choice == "Brazil" : 
            country =  22
        elif choice ==" Brunei Darussalam" : 
            country =  23
        elif choice == "Bulgaria" : 
            country =  24
        elif choice == "Burkina Faso" : 
             country =  25
        elif choice == "Burundi" : 
             country =  26
        elif choice == "Côte d'Ivoire" : 
             country =  27
        elif choice == "Cabo Verde" : 
             country =  28
        elif choice == "Cambodia" : 
             country =  29
        elif choice == "Cameroon" : 
             country =  30
        elif choice == "Canada" : 
             country =  31
        elif choice == "Central African Republic" : 
             country =  32
        elif choice == "Chad" : 
             country =  33
        elif choice == "Chile" : 
             country =  34
        elif choice == "China" : 
             country =  35
        elif choice == "Colombia" : 
             country =  36
        elif choice == "Comoros" : 
             country =  37
        elif choice == "Congo" : 
             country =  38
        elif choice == "Costa Rica" : 
             country =  39
        elif choice == "Croatia" : 
             country =  40
        elif choice == "Cuba" : 
             country =  41
        elif choice == "Cyprus" : 
             country =  42
        elif choice == "Czechia" : 
             country =  43
        elif choice == "Democratic People's Republic of Korea" : 
             country =  44
        elif choice == "Democratic Republic of the Congo" : 
             country =  45
        elif choice == "Denmark" : 
             country =  46
        elif choice == "Djibouti" : 
             country =  47
        elif choice == "Dominican Republic" : 
             country =  48
        elif choice == "Ecuador" : 
             country =  49
        elif choice == "Egypt" : 
             country =  50
        elif choice == "El Salvador" : 
             country =  51
        elif choice == "Equatorial Guinea" : 
             country =  52
        elif choice == "Eritrea" : 
             country =  53
        elif choice == "Estonia" : 
             country =  54
        elif choice == "Ethiopia" : 
             country =  55
        elif choice == "Fiji" : 
             country =  56
        elif choice == "Finland" : 
             country =  57
        elif choice == "France" : 
             country =  58
        elif choice == "Gabon" : 
             country =  59
        elif choice == "Gambia" : 
             country =  60
        elif choice == "Georgia" : 
             country =  61
        elif choice == "Germany" : 
             country =  62
        elif choice == "Ghana" : 
             country =  63
        elif choice == "Greece" : 
             country =  64
        elif choice == "Grenada" : 
             country =  65
        elif choice == "Guatemala" : 
             country =  66
        elif choice == "Guinea" : 
             country =  67
        elif choice == "Guinea-Bissau" : 
             country =  68
        elif choice == "Guyana" : 
             country =  69
        elif choice == "Haiti" : 
             country =  70
        elif choice == "Honduras" : 
             country =  71
        elif choice == "Hungary" : 
             country =  72
        elif choice == "Iceland" : 
             country =  73
        elif choice == "India" : 
             country =  74
        elif choice == "Indonesia" : 
             country =  75
        elif choice == "Iran (Islamic Republic of)" : 
             country =  76
        elif choice == "Iraq" : 
             country =  77
        elif choice == "Ireland" : 
             country =  78
        elif choice == "Israel" : 
             country =  79
        elif choice == "Italy" : 
             country =  80
        elif choice == "Jamaica" : 
             country =  81
        elif choice == "Japan" : 
             country =  82
        elif choice == "Jordan" : 
             country =  83
        elif choice == "Kazakhstan" : 
             country =  84
        elif choice == "Kenya" : 
             country =  85
        elif choice == "Kiribati" : 
             country =  86
        elif choice == "Kuwait" : 
             country =  87
        elif choice == "Kyrgyzstan" : 
             country =  88
        elif choice == "Lao People's Democratic Republic" : 
             country =  89
        elif choice == "Latvia" : 
             country =  90
        elif choice == "Lebanon" : 
             country =  91
        elif choice == "Lesotho" : 
             country =  92
        elif choice == "Liberia" : 
             country =  93
        elif choice == "Libya" : 
             country =  94
        elif choice == "Lithuania" : 
             country =  95
        elif choice == "Luxembourg" : 
             country =  96
        elif choice == "Madagascar" : 
             country =  97
        elif choice == "Malawi" : 
             country =  98
        elif choice == "Malaysia" : 
             country =  99
        elif choice == "Maldives" : 
             country =  100
        elif choice == "Mali" : 
             country =  101
        elif choice == "Malta" : 
             country =  102
        elif choice == "Mauritania" : 
             country =  103
        elif choice == "Mauritius" : 
             country =  104
        elif choice == "Mexico" : 
             country =  105
        elif choice == "Micronesia (Federated States of)" : 
             country =  106
        elif choice == "Mongolia" : 
             country =  107
        elif choice == "Montenegro" : 
             country =  108
        elif choice == "Morocco" : 
             country =  109
        elif choice == "Mozambique" : 
             country =  110
        elif choice == "Myanmar" : 
             country =  111
        elif choice == "Namibia" : 
             country =  112
        elif choice == "Nepal" : 
             country =  113
        elif choice == "Netherlands" : 
             country =  114
        elif choice == "New Zealand" : 
             country =  115
        elif choice == "Nicaragua" : 
             country =  116
        elif choice == "Niger" : 
             country =  117
        elif choice == "Nigeria" : 
             country =  118
        elif choice == "Norway" : 
             country =  119
        elif choice == "Oman" : 
             country =  120
        elif choice == "Pakistan" : 
             country =  121
        elif choice == "Panama" : 
             country =  122
        elif choice == "Papua New Guinea" : 
             country =  123
        elif choice == "Paraguay" : 
             country =  124
        elif choice == "Peru" : 
             country =  125
        elif choice == "Philippines" : 
             country =  126
        elif choice == "Poland" : 
             country =  127
        elif choice == "Portugal" : 
             country =  128
        elif choice == "Qatar" : 
             country =  129
        elif choice == "Republic of Korea" : 
             country =  130
        elif choice == "Republic of Moldova" : 
             country =  131
        elif choice == "Romania ": 
             country =  132
        elif choice == "Russian Federation" : 
             country =  133
        elif choice == "Rwanda" : 
             country =  134
        elif choice == "Saint Lucia" : 
             country =  135
        elif choice == "Saint Vincent and the Grenadines" : 
             country =  136
        elif choice == "Samoa" : 
             country =  137
        elif choice == "Sao Tome and Principe" : 
             country =  138
        elif choice == "Saudi Arabia" : 
             country =  139
        elif choice == "Senegal" : 
             country =  140
        elif choice == "Serbia" : 
             country =  141
        elif choice == "Seychelles" : 
             country =  142
        elif choice == "Sierra Leone" : 
             country =  143
        elif choice == "Singapore" : 
             country =  144
        elif choice == "Slovakia" : 
             country =  145
        elif choice == "Slovenia" : 
             country =  146
        elif choice == "Solomon Islands" : 
             country =  147
        elif choice == "Somalia" : 
             country =  148
        elif choice == "South Africa" : 
             country =  149
        elif choice == "South Sudan" : 
             country =  150
        elif choice == "Spain" : 
             country =  151
        elif choice == "Sri Lanka" : 
             country =  152
        elif choice == "Sudan" : 
             country =  153
        elif choice == "Suriname" : 
             country =  154
        elif choice == "Swaziland" : 
             country =  155
        elif choice == "Sweden" : 
             country =  156
        elif choice == "Switzerland" : 
             country =  157
        elif choice == "Syrian Arab Republic" : 
             country =  158
        elif choice == "Tajikistan" : 
             country =  159
        elif choice == "Thailand" : 
             country =  160
        elif choice == "The former Yugoslav republic of Macedonia" : 
             country =  161
        elif choice == "Timor-Leste" : 
             country =  162
        elif choice == "Togo" : 
             country =  163
        elif choice == "Tonga" : 
             country =  164
        elif choice == "Trinidad and Tobago" : 
             country =  165
        elif choice == "Tunisia" : 
             country =  166
        elif choice == "Turkey" : 
             country =  167
        elif choice == "Turkmenistan" : 
             country =  168
        elif choice == "Uganda" : 
             country =  169
        elif choice == "Ukraine" : 
             country =  170
        elif choice == "United Arab Emirates" : 
             country =  171
        elif choice == "United Kingdom of Great Britain and Northern Ireland" : 
             country =  172
        elif choice == "United Republic of Tanzania" : 
             country =  173
        elif choice == "United States of America" : 
             country =  174
        elif choice == "Uruguay" : 
             country =  175
        elif choice == "Uzbekistan" : 
             country =  176
        elif choice == "Vanuatu" : 
             country =  177
        elif choice == "Venezuela (Bolivarian Republic of)" : 
             country =  178
        elif choice == "Viet Nam" : 
             country =  179
        elif choice == "Yemen" : 
             country =  180
        elif choice == "Zambia" : 
             country =  181
        elif choice == "Zimbabwe" : 
             country =  182
        else:
            st.markdown("Not selected within the given 193 countries")

    if st.subheader("Year"):
        year = st.number_input('Year')

    if st.subheader("Economic Status of the Country"):
        data_dim = st.radio("Show Dimension By ", ("Developing", "Developed"))
        if data_dim == 'Developing':
            status = 1
        elif data_dim == 'Developed':
            status = 2
        else:
            st.markdown("Not selected within the given status")

    if st.subheader("Adult Mortality Rates"):
        adult_mortality = st.number_input('Adult Mortality Rates of both genders: probability of dying between 15 and 60 years per 1000 population.')

    if st.subheader("Number of Infant Deaths"):
        infant_deaths = st.number_input('Number of Infant Deaths per 1000 population. Give an integer')

    if st.subheader("Alcohol Consumption Rate"):
        alcohol = st.number_input('Per capita (age: 15+) consumption (in litres of pure alcohol)')

    if st.subheader("Expense on Healthcare"):
        percentage_expenditure = st.number_input('Expenditure on health as a percentage of Gross Domestic Product per capita(%)')

    if st.subheader("Hepatitis-B Immunization Coverage Rate"):
        hepatitis_b = st.number_input('Hepatitis B (HepB) immunization coverage among 1-year-olds (%)')

    if st.subheader("Measeles: Number of Reported Cases"):
        measles = st.number_input('Measles - number of reported cases per 1000 population. Enter an intger')  

    if st.subheader("Average BMI"):
        bmi = st.number_input('Average Body Mass Index of entire population') 

    if st.subheader("Below 5 yo Child Death Rate"):
        under_five_deaths = st.number_input('Number of under-five deaths per 1000 population. Enter an integer')
    
    if st.subheader("(Pol3) Immunization Coverage Rate"):
        polio = st.number_input('Polio (Pol3) immunization coverage among 1-year-olds (%)')
    
    if st.subheader("Government's Healthcare Expense Rate"):
        total_expenditure= st.number_input('General government expenditure on health as a percentage of total government expenditure (%)')
    
    if st.subheader("(DTP3) Immunization Coverage Rate"):
        diphtheria = st.number_input('Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)')  
    
    if st.subheader("HIV/AIDS Deaths"):
        hiv_aids = st.number_input('Deaths per 1000 live births HIV/AIDS (0-4 years)') 

    if st.subheader("GDP"):
        gdp = st.number_input('Gross Domestic Product per capita (in USD)')

    if st.subheader("Population"):
        population = st.number_input('Population of the country')
    
    if st.subheader("Thinnness Percentage (10-19 yo)"):
        thinness_1_19_years = st.number_input('Prevalence of thinness among children and adolescents for Age 10 to 19 (%)')
    
    if st.subheader("Thinnness Percentage of Children (5-9 yo)"):
        thinness_5_9_years = st.number_input('Prevalence of thinness among children for Age 5 to 9 (%)')
    
    if st.subheader("Human Development Index"):
        income_composition_of_resources = st.number_input('Human Development Index in terms of income composition of resources (index ranging from 0 to 1)')
    
    if st.subheader("Schooling Years"):
        schooling = st.number_input('Number of years of Schooling(years)')


    feat = [country, year, status, adult_mortality, infant_deaths, alcohol,
    percentage_expenditure, hepatitis_b, measles, bmi, under_five_deaths, polio,
    total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years, thinness_5_9_years, income_composition_of_resources, schooling]

    data = {'country': country, 
            'year': year, 
            'status': status, 
            'adult_mortality': adult_mortality, 
            'infant_deaths': infant_deaths, 
            'alcohol': alcohol, 
            'percentage_expenditure': percentage_expenditure, 
            'hepatitis_b': hepatitis_b, 
            'measles': measles,
            'bmi': bmi,
            'under_five_deaths': under_five_deaths,
            'polio': polio,
            'total_expenditure': total_expenditure,
            'diphtheria': diphtheria,
            'hiv_aids': hiv_aids,
            'gdp': gdp,
            'population': population, 
            'thinness_1_19_years': thinness_1_19_years, 
            'thinness_5_9_years': thinness_5_9_years,
            'income_composition_of_resources': income_composition_of_resources,
            'schooling': schooling} 
    
    df = pd.DataFrame(data, index=[0])

    st.subheader('User Input parameters')
    st.write(df)

    feat = [(x) for x in feat]
    final_features = [np.array(feat)]
    
    #-----------------------------------------------------------------------------------
    
    import pandas as pd
    import numpy as np

    df = pd.read_csv('Preprocessed_Life_Expectancy_Data.csv')
    df = df. drop('Unnamed: 0', axis=1)

    X = df.drop('life_expectancy', axis = 1) 
    y = df['life_expectancy']

    from sklearn import ensemble
    extra = ensemble.ExtraTreesRegressor(random_state = 42, bootstrap = False, max_depth = 890, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 5, n_estimators = 1400)
    extra.fit(X, y)

    # -----------------------------------------------------------------------------------

    # model = pickle.load(open('ExtraTrees_Life_Expectancy.pkl', 'rb'))  # get the model

    st.subheader('   ')

    if st.button('Predict'):
       #prediction = model.predict(final_features)
       prediction = extra.predict(final_features)
       st.success(f'Predicted life expectancy is : {round(prediction[0], 1)} years')

elif choice == "Model Performance Metrics":
    st.subheader("Model Performance Metrics")
    st.info('Mean absolute error (MAE): 1.0564712470881465') 
    st.info('Mean squared error (MSE): 3.2797066467789717')
    st.info('Root mean squared error (RMSE): 1.8109960372068659')
    st.info('R Squared: 0.9648817803845204')
    st.info('Adjusted R Squared: 0.9640212405806405')
    st.info('Accuracy (Computed from Actual & Predicted-- please see code**): 98.368%')

else:
    st.markdown('  ')
    



