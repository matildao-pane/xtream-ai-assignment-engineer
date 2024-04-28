model = XGBRegressor(seed=RAND_STATE, objective='reg:squarederror', colsample_bytree= 0.9, 
                     learning_rate= 0.05, max_depth= 5, n_estimators=250, subsample= 0.9), 

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print('\nResults:')
results[name] = {'RMSE': rmse, 'R2 Score': r2, 'MAE': mae, 'Training Time': training_time, 'best_params':best_params}
print(f'{name} - \nRMSE: {rmse}, \nR2 Score: {r2}, \nMAE: {mae}, \nTraining Time: {training_time} seconds, \nbest_params {best_params}' )
plot(y_test, y_pred_linear,name)

plot_feature_importance(model, FEAT, name )

# Save the model using pickle
with open(f'{name}_.pkl', 'wb') as f:
    pickle.dump(model, f)
