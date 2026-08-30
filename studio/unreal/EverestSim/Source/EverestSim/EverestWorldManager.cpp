#include "EverestWorldManager.h"
#include "EverestTerrainActor.h"
#include "EverestSnowPatchActor.h"
#include "EverestRobotActor.h"
#include "EverestSnowfallActor.h"
#include "WebSocketsModule.h"
#include "Modules/ModuleManager.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonReader.h"
#include "Dom/JsonObject.h"
#include "Engine/World.h"
#include "Engine/DirectionalLight.h"
#include "Engine/SkyLight.h"
#include "Engine/ExponentialHeightFog.h"
#include "Engine/SkyAtmosphere.h"
#include "Components/LightComponent.h"
#include "Components/ExponentialHeightFogComponent.h"

AEverestWorldManager::AEverestWorldManager()
{
    PrimaryActorTick.bCanEverTick = false;
}

void AEverestWorldManager::BeginPlay()
{
    Super::BeginPlay();

    MacroTerrain = GetWorld()->SpawnActor<AEverestTerrainActor>();
    MacroTerrain->AddActorWorldOffset(FVector(0, 0, -12.f));
    Terrain = GetWorld()->SpawnActor<AEverestTerrainActor>();
    Snow = GetWorld()->SpawnActor<AEverestSnowPatchActor>();
    Robot = GetWorld()->SpawnActor<AEverestRobotActor>();
    Snowfall = GetWorld()->SpawnActor<AEverestSnowfallActor>();

    ADirectionalLight* Sun = GetWorld()->SpawnActor<ADirectionalLight>();
    Sun->SetActorRotation(FRotator(-28.f, -32.f, 0.f));
    Sun->GetLightComponent()->SetIntensity(7.5f);

    ASkyLight* Sky = GetWorld()->SpawnActor<ASkyLight>();
    Sky->GetLightComponent()->SetIntensity(0.9f);
    GetWorld()->SpawnActor<ASkyAtmosphere>();

    AExponentialHeightFog* Fog = GetWorld()->SpawnActor<AExponentialHeightFog>();
    Fog->GetComponent()->SetFogDensity(0.0018f);
    Fog->GetComponent()->SetFogHeightFalloff(0.18f);
    Fog->GetComponent()->SetVolumetricFog(true);

    FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
    Socket = FWebSocketsModule::Get().CreateWebSocket(TEXT("ws://127.0.0.1:8765"));
    Socket->OnMessage().AddUObject(this, &AEverestWorldManager::OnMessage);
    Socket->OnConnected().AddLambda([]()
    {
        UE_LOG(LogTemp, Display, TEXT("Everest physics bridge connected"));
    });
    Socket->OnConnectionError().AddLambda([](const FString& Error)
    {
        UE_LOG(LogTemp, Warning, TEXT("Everest bridge: %s"), *Error);
    });
    Socket->Connect();
}

void AEverestWorldManager::OnMessage(const FString& Message)
{
    TSharedPtr<FJsonObject> RootObject;
    const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Message);
    if (!FJsonSerializer::Deserialize(Reader, RootObject) || !RootObject.IsValid())
    {
        return;
    }

    FString Type;
    if (!RootObject->TryGetStringField(TEXT("type"), Type))
    {
        return;
    }

    const TSharedPtr<FJsonObject>* DataPtr = nullptr;
    if (!RootObject->TryGetObjectField(TEXT("data"), DataPtr) || !DataPtr || !DataPtr->IsValid())
    {
        return;
    }
    const TSharedPtr<FJsonObject> Data = *DataPtr;

    if (Type == TEXT("scene") && Robot)
    {
        Robot->ApplyScene(Data);
    }
    else if (Type == TEXT("macro_terrain") && MacroTerrain)
    {
        MacroTerrain->ApplyTerrain(Data);
    }
    else if (Type == TEXT("terrain") && Terrain)
    {
        Terrain->ApplyTerrain(Data);
    }
    else if (Type == TEXT("snow") && Snow)
    {
        Snow->ApplySnowFrame(Data);
    }
    else if (Type == TEXT("frame") && Robot)
    {
        Robot->ApplyFrame(Data);
        if (!Snowfall)
        {
            return;
        }

        Snowfall->SetFocus(Robot->GetPelvisLocation());
        double Rate = 0.0;
        double Wind = 0.0;
        double Direction = 0.0;
        double Visibility = 1.0;

        const TSharedPtr<FJsonObject>* SnowObject = nullptr;
        if (Data->TryGetObjectField(TEXT("snow"), SnowObject) && SnowObject && SnowObject->IsValid())
        {
            (*SnowObject)->TryGetNumberField(TEXT("snowfall_mm_h"), Rate);
            (*SnowObject)->TryGetNumberField(TEXT("wind_speed_m_s"), Wind);
            (*SnowObject)->TryGetNumberField(TEXT("wind_direction_deg"), Direction);
        }

        const TSharedPtr<FJsonObject>* WeatherParameters = nullptr;
        if (Data->TryGetObjectField(TEXT("weather_parameters"), WeatherParameters) && WeatherParameters && WeatherParameters->IsValid())
        {
            (*WeatherParameters)->TryGetNumberField(TEXT("visibility_scale"), Visibility);
        }
        Snowfall->SetWeather(static_cast<float>(Rate), static_cast<float>(Wind), static_cast<float>(Direction), static_cast<float>(Visibility));
    }
}

void AEverestWorldManager::SendControl(const FString& Action, const FString& JsonValue)
{
    if (!Socket.IsValid() || !Socket->IsConnected())
    {
        return;
    }
    const FString Payload = FString::Printf(
        TEXT("{\"type\":\"control\",\"action\":\"%s\",\"value\":%s}"),
        *Action,
        *JsonValue);
    Socket->Send(Payload);
}

void AEverestWorldManager::SendRobotCommand(float Forward, float Lateral, float YawRate)
{
    SendControl(TEXT("command"), FString::Printf(TEXT("[%.5f,%.5f,%.5f]"), Forward, Lateral, YawRate));
}

void AEverestWorldManager::SetSimulationPaused(bool bPaused)
{
    SendControl(TEXT("pause"), bPaused ? TEXT("true") : TEXT("false"));
}
