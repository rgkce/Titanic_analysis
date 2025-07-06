# 🚢 EDA Nedir? Titanic Veri Seti Üzerinden Adım Adım Keşifsel Veri Analizi

---

## 👋 Giriş

Makine öğrenmesi ve veri bilimi projelerinde başarıya ulaşmanın ilk adımı **veriyi iyi tanımaktır**. Bu noktada **EDA** (Exploratory Data Analysis – Keşifsel Veri Analizi) devreye girer. Bu yazıda, klasik bir örnek olan **Titanic veri seti** üzerinden EDA sürecini adım adım uygulayıp yorumluyorum.

---

## 📦 Veri Setini Tanıyalım

**Titanic veri seti**, 1912 yılında batan Titanic gemisinin yolcularına ait bilgileri içerir. Veri setinde yolcuların yaşı, cinsiyeti, bilet sınıfı gibi bilgiler ve hayatta kalıp kalmadığı yer alır.

**İlk tespitler:**

- `Age` sütununda eksik değerler vardı → Ortanca (median) değerle dolduruldu.
- `Embarked` sütunundaki eksikler → En sık görülen değerle dolduruldu.
- `Cabin` sütunu neredeyse tamamen eksik → Veri setinden çıkarıldı.

```python
import pandas as pd

df = pd.read_csv("titanic.csv")

# Eksik değerlerin doldurulması
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.drop('Cabin', axis=1, inplace=True)

df.head()

```

## 📊 Veri Görselleştirme

Veriyi anlamak için Seaborn ve Matplotlib kullanarak çeşitli grafikler çizdim.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Hayatta Kalma Durumu
sns.countplot(data=df, x='Survived')
plt.title("Hayatta Kalma Dağılımı")
plt.show()
#Yorum: Hayatta kalanlar (1) ve kalamayanlar (0) arasında ciddi fark var. Kurtulamayanların sayısı daha fazla.


#2️⃣ Cinsiyete Göre Hayatta Kalma
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title("Cinsiyete Göre Hayatta Kalma")
plt.show()
#Yorum: Kadınların hayatta kalma oranı erkeklere göre oldukça yüksek. Bu, gemide kadınlara öncelik verildiğini gösteriyor olabilir.


#3️⃣ Sınıfa Göre Hayatta Kalma
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title("Sınıfa Göre Hayatta Kalma")
plt.show()
#Yorum: 1. sınıfta yolculuk yapanların hayatta kalma oranı daha yüksek. 3. sınıfta bu oran oldukça düşük.

#4️⃣ Cinsiyet + Sınıf Pivot Table
pivot = df.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')
print(pivot)
#Yorum: Örneğin 1. sınıf kadınların %96’sı hayatta kalmış. 3. sınıfta ise erkeklerin kurtulma şansı çok düşük.

#5️⃣ Yaş Dağılımı & Hayatta Kalma
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title("Yaş Dağılımı ve Hayatta Kalma")
plt.show()
#Yorum: 0–10 yaş arası çocuklarda hayatta kalma oranı yüksek. Yaş arttıkça hayatta kalma oranı düşüyor.

#6️⃣ Korelasyon Matrisi
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Korelasyon Matrisi")
plt.show()
#Yorum: Cinsiyet, sınıf ve ücret (Fare) gibi değişkenlerin hayatta kalma ile korelasyonu güçlü.

#7️⃣ Aykırı Değer Analizi (Boxplot)
sns.boxplot(x=df['Age'])
plt.title("Age için Boxplot")
plt.show()
#Yorum: Yaş sütununda bazı uç değerler mevcut ama genel dağılım dengeli görünüyor.
```

## 💡 Çıkarımlar

Kadınların hayatta kalma oranı erkeklerden çok daha yüksek.

- Üst sınıfta yolculuk yapanlar daha avantajlıydı.

- Küçük yaştaki yolcular (çocuklar) için öncelik verilmiş olabilir.

- Sınıf ve cinsiyet gibi kategorik değişkenler, hayatta kalma durumu üzerinde belirgin etkiye sahip.

## ✍️ Kapanış

Bu çalışma ile veriyi temizleme, görselleştirme ve yorumlama pratiklerini gerçek bir veri seti üzerinde uyguladım. EDA, veri biliminin en önemli adımlarından biridir. Titanic örneği de bunu en iyi şekilde gösteriyor.

Siz de kendi projelerinizde benzer adımları izleyerek verinizi daha iyi anlayabilir, modelleme aşamasına sağlam bir temel oluşturabilirsiniz.

## 📢 Bağlantı

LinkedIn: https://www.linkedin.com/in/rümeysa-gökçe/

GitHub: https://github.com/rgkce

Beğendiyseniz destek olmayı ve yorum bırakmayı unutmayın! 🚀
