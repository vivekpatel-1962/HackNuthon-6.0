# Generated by Django 5.1.7 on 2025-03-30 04:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0003_bankaccount'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bankaccount',
            name='current_balance',
            field=models.DecimalField(decimal_places=2, default=1000000, max_digits=12),
        ),
    ]
