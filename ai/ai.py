import pickle
from typing import Any 
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from dataclasses import dataclass

from dbcore.models import TrainedModelFile


@dataclass
class AIModel:
    data: dict
    pd_dields = [
        'gndr', 'age', 'brth_yr', 
        'city', 'accnt_pnsn_schm', 'year_oper',
        'quarter', 'enter_year', 
        'years_without_contrib', 'region',
    ]
    predict_info = {
        0: ['девушка, из деревни, 27-38 лет, кол-во операций ~8,  сумма взносов ~19242', 'кол-во операций 11-13,  сумма взносов 6153-35638'],
        1: ['девушка, из деревни, 32-59 лет, кол-во операций ~6,  сумма взносов ~27986', 'кол-во операций 12-13,  сумма взносов 16373-48908'],
        2: ['мужчины, из деревни, 45-53 лет, кол-во операций ~9,  сумма взносов ~29604', 'кол-во операций 11-13,  сумма взносов 13021-43364'],
        3: ['девушка, из деревни, 30-40 лет, кол-во операций ~10,  сумма взносов ~25650', 'кол-во операций 12-13,  сумма взносов 7283-44155'],
    }

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        to_model_data = self._parse_data()
        df = pd.DataFrame(
            data=[to_model_data, to_model_data], 
            columns=[
                'gndr', 'age', 'brth_yr', 'city', 'accnt_pnsn_schm', 'year_oper',
                'quarter', 'Возраст вступления в фонд', 'Сколько лет не делал взносы', 'Регион',
                'coef', 'coef_dollar', 'coef_neft', 'coef_gaz'
            ]
        )   

        predict = self._model(data=df)

        if predict:
            return self.predict_info[predict]
        
        return ['Модели нет', 'Необходимо добавить модель через админ-панель']

    def _parse_data(self) -> list:
        data = []
        # data.append(self.data['gender_mars'] if self.data['gender_mars'] else self.data['gender_venus'])
        # data.append(self.data['gender_mars'] if self.data['gender_mars'] else self.data['gender_venus'])
        # data.append(self.data[key])
        for key in self.pd_dields:
            if key == 'gndr':
                data.append(0 if self.data['gender_mars'] else 1)
            elif key == 'city':
                data.append(1 if self.data['city_true'] else 0)
            else:
                data.append(self.data[key])

        data.append(0.19720)
        data.append(28.46360)
        data.append(4.95860)
        data.append(28.46360)

        return data
    
    def _model(self, data: list):
        #подключения модели для преобразования 
        np.seterr(divide='ignore', invalid='ignore')
        pca = PCA(n_components=2)
        scaler = StandardScaler()
        pd_dields = [
            'gndr', 'age', 'brth_yr', 'city', 'accnt_pnsn_schm', 'year_oper',
            'quarter', 'Возраст вступления в фонд', 'Сколько лет не делал взносы', 'Регион',
            'coef', 'coef_dollar', 'coef_neft', 'coef_gaz'
        ]
        #скалирование и нормализация
        x_scaled = scaler.fit_transform(data)
        x_normal = normalize(x_scaled)
        x_normal = pd.DataFrame(x_normal)

        #преобразование данные в 2-х мерные
        x_principal = pca.fit_transform(x_normal)
        x_principal = pd.DataFrame(x_principal)
        x_principal.columns = ['V1', 'V2']

        #загрузка модели
        trained_model = TrainedModelFile.objects.last()
        if trained_model:
            with open(f'media/{trained_model.file}', 'rb') as file: 
                pickle_model = pickle.load(file)
            #предсказание
            return pickle_model.predict([x_principal.iloc[0, :2]])[0]
        
        return None