<!DOCTYPE html>
<html lang="en">
<head>
  <link rel= "Styles" href="Styles.css">
</head>
<body>
    <header>
        <h1>Supervised Learning Techniques</h1>
    </header>
    <main>
      <p>
        This project explores supervised learning techniques, specifically focusing on Multiclass Logistic Regression and K-Nearest Neighbor (KNN) algorithms. The goal is to classify penguin species based on features such as bill length, bill depth, and flipper length.
      </p>
      <h2>
        Implementation
      </h2>
      <p>
        The implementation involves the following steps
        <ul>
          <li>Data preprocessing and splitting into training and testing sets</li>
          <li>Application of Multiclass Logistic Regression and KNN algorithms to classify penguin species.</li>
          <li>Evaluation of model performance using metrics such as accuracy, precision, and F1 score.</li>
          <li>Optimization of hyperparameters to improve model performance</li>
        </ul>
      </p>
      <h2>
        Results
      </h2>
      <p>
        The results of the models before and after hyperparameter optimization are as follows:
        <ul>
          <li>
            <h3>
              Logistic Regression
            </h3>
            <ul>
              <li>
                Before optimization: Accuracy = 93%, Precision = 93.54%, F1 Score = 92.97%
              </li>
              <li>
                After optimization: Accuracy = 96%, Precision = 96.307%, F1 Score = 95.97%
              </li>
            </ul>
          </li>
          <li>
            <h3>
              K-Nearest Neighbor
            </h3>
            <ul>
              <li>
                Before optimization: Accuracy = 97%, Precision = 97.18%, F1 Score = 96.94%
              </li>
              <li>
                After optimization: Accuracy = 99%, Precision = 99.02041%, F1 Score = 98.99404%
              </li>
            </ul>
          </li>
        </ul>
      </p>
      <h2>
        Conclusion
      </h2>
      <p>
        The K-Nearest Neighbor algorithm, especially after hyperparameter optimization, outperformed Multiclass Logistic Regression in classifying penguin species. This project demonstrates the importance of model selection and hyperparameter tuning in supervised learning tasks.
      </p>
    </main>
    <footer>
        <p>&copy; 2024 Dewald Oosthuizen (38336529)</p>
    </footer>
</body>
</html>
