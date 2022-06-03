# BGNBD & GG ile CLTV Tahmini ve Sonuçların Uzak Sunucuya Gönderilmesi
###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.
#
# Buna yönelik olarak müşterilerin davranışlarını tanımlayacağız ve
# bu davranışlarda öbeklenmelere göre gruplar oluşturacağız.
#
# Yani ortak davranışlar sergileyenleri aynı gruplara alacağız ve
# bu gruplara özel satış ve pazarlama teknikleri geliştirmeye çalışacağız.

# Veri Seti Hikayesi
#
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
#
# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
#
# Bu şirket hediyelik eşya satıyor. Promosyon ürünleri gibi düşünebilir.
#
# Müşterilerinin çoğu da toptancı.
#
# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


#pip install lifetimes
#pip install sqlalchemy
#conda install -c anaconda mysql-connector-python
#conda install -c conda-forge mysql


from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Verinin Excel'den Okunması

df_ = pd.read_excel(r"C:\Users\Casper\PycharmProjects\DataScience\WEEK03\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()

df.shape #Satır ve sütun sayısı
df.head()
df.describe().T


# Veri Ön İşleme

df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Lifetime Veri Yapısının Hazırlanması
# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
# İndex düzenlemesi yapar
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head()

# satın alma başına ortalama kazanç
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
cltv_df.head()

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# cltv_df = cltv_df[cltv_df["monetary"] > 0]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

#f ile ilgili tip hatası alırsak;
# cltv_df["frequency"] = cltv_df["frequency"].astype(int)


# BG-NBD Modelinin Kurulması

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

# GAMMA-GAMMA Modelinin Kurulması

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])


# BG-NBD ve GG modeli ile CLTV'nin hesaplanması
######################################################

#Görev 1:6 aylık CLTV Predictio

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
cltv.shape

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

# recency ve T değerlerinin yakın olması durumunda cltv değerleri yüksektir.
# recency ve T değerlerinin uzak olması durumunda cltv değeri düşükütür
# monetary değerinin büyük olması durumunda cltv değeri içinde büyüktür diyebiliriz.

########################################################################################

#Görev 2:Farklı zaman periyotlarından oluşan CLTV analiz

# 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# 1 aylık tahmin

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

#12 aylık tahmin

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

#1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_1_month", ascending=False).head(10)

# cltv değeri frequency ile doğru orantılı bir şekilde artış göstermektedir.
###############################################################################################################
# 12 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz

bgf.predict(4*12,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_12_month"] = bgf.predict(4*12,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.sort_values("expected_purc_12_month", ascending=False).head(10)

# 12 aylık cltv değerinde yaklaşık olarak 12 kat bir artış olmaktadır.

######################################################################################

#Görev 3: Segmentasyon ve Aksiyon Önerileri

cltv["segment"] = pd.qcut(cltv["clv"], 4, labels=["D", "C", "B", "A"])
cltv.head(20)


cltv.sort_values(by="clv", ascending=False).head(20)


cltv.groupby("segment").agg(
    {"count", "mean", "sum"})

# 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz

# A segmenti için şampiyonlar diyebiliriz cltv toplam değeri çok yüksek bu müşterileri kaybetmemek lazım .
# D segmenti için aynı şey söz konusu değil  cltv toplam değeri çok düşük yani kaybetmeye en yakın olunan segment .
# Bu segmentte yer alanlara indirim mesajları, kargo ücretsiz gibi sürekli bildirimler gönderilerek alışveriş yapılması
# sağlanabilir.

############################################################################################################

#Görev 4: Veri tabanına kayıt gönderme

# Verinin Veri Tabanından Okunması
######################################

# credentials.
creds = {'user': '***',
         'passwd': '***',
         'host': '***',
         'port': ***,
         'db': '***'}

# MySQL conection string.
connstr = '***'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

# conn.close()


pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)



pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)


retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()
# df = retail_mysql_df.copy()
