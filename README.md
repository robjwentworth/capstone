Auto Claim Customer Complaint Web Application.

The Auto Claim Customer Complaint Web Application determines the probability that a specific customer has of having one or more complaints against the company. The application looks at six specific points of information: Gender, Income, Monthly Premium, Number of Months since their last claim, Generalized Location, and a Generalized size of their vehicle to make the determination.

The Application is motivated by asking a business specific question. In this case, the question I wanted to answer centered around insurance. I have had a career in the insurance field for the past 9 years and carry domain knowledge in that domain. I found the data set around Auto Claims at a customer level of detail and decided to answer a question in that domain. The number of complaints field intrigued me and seemed to be an ideal candidate for further study.

The Application is written in Python 2.7.12 and HTML Bootstrap. The interface is fairly simple. The user inputs the information, hits Submit and a probability is returned. The probability is returned by taking the information and running it up against a Gradient Boosting Classifier model that has been pickled to store on site. The site is hosted by Amazon EC2.

The code repository can be found on github here: https://github.com/robjwentworth/capstone

The application can be accessed at http://www.robjwentworth.com
